import argparse
import json
import logging
import os
import random
import yaml
import numpy as np
import cv2
from PIL import Image
import duckdb
import base64
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lightning.pytorch as pl
from vocaludf import mlp
import torchmetrics
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict
from vocaludf.utils import parse_signature
from vocaludf.udf_proposer import UDFProposer
import string
import importlib

client = OpenAI()

logging.basicConfig()
logger = logging.getLogger("vocal_udf")
logger.setLevel(logging.DEBUG)

class CustomImageDataset(Dataset):
    def __init__(self, data, train):
        self.X = [d["image_features"] for d in data]
        if train:
            self.y = [d["llm_label"] for d in data]
        else:
            self.y = [d["label"] for d in data]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ModelDistiller(UDFProposer):
    def __init__(self, config, prompt_config, dataset, udf_signature, udf_description, gt_udf_name, run_id, n_train, save_labeled_data, load_labeled_data):
        self.config = config
        self.prompt_config = prompt_config
        self.dataset = dataset
        udf_name, udf_vars = parse_signature(udf_signature)
        self.udf_class = udf_name.lower()
        self.n_obj = len(udf_vars)
        assert self.n_obj in [1, 2], "n_obj must be 1 or 2"
        self.udf_description = udf_description
        self.run_id = run_id
        self.n_train = n_train
        self.n_test = 1000
        self.save_labeled_data = save_labeled_data
        self.load_labeled_data = load_labeled_data
        # self.n_train = config["model_distiller"]["n_train"]

        module_name, function_name = gt_udf_name.split(".")
        module_name = "udfs.{}".format(module_name)
        module = importlib.import_module(module_name)
        self.gt_udf = getattr(module, function_name)

        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )

        # NOTE: LLM doesn't generate labels in some cases, so we need to double the number of samples (i.e., self.n_train * 2) to ensure we have enough training samples
        self.df_train, self.df_test = self.construct_train_and_test_data(self.n_obj, self.n_train * 2, self.n_test)

        # Load the CLIP model
        # model_name = "openai/clip-vit-base-patch32"
        model_name = os.path.join(self.config['model_dir'], 'clip-vit-base-patch32')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # self.clip_model.save_pretrained(os.path.join(self.config['model_dir'], 'clip-vit-base-patch32'))
        # self.processor.save_pretrained(os.path.join(self.config['model_dir'], 'clip-vit-base-patch32'))

    def frame_processing(self, row):
        vid = row.vid if self.n_obj == 1 else row.o1_vid
        fid = row.fid if self.n_obj == 1 else row.o1_fid
        cap = cv2.VideoCapture(
            os.path.join(
                self.config['data_dir'],
                self.dataset,
                f'video_{str(vid//1000*1000).zfill(5)}-{str((vid//1000+1)*1000).zfill(5)}',
                f"video_{str(vid).zfill(5)}.mp4"
            )
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            logger.debug("Failed to read the frame")
            return None
        cap.release()
        image_size = frame.shape[:2]
        if self.n_obj == 1:
            x1, y1, x2, y2 = self.expand_box(row.x1, row.y1, row.x2, row.y2, image_size)
            frame = frame[y1:y2, x1:x2]
        else:
            o1_x1, o1_y1, o1_x2, o1_y2 = self.expand_box(row.o1_x1, row.o1_y1, row.o1_x2, row.o1_y2, image_size)
            o2_x1, o2_y1, o2_x2, o2_y2 = self.expand_box(row.o2_x1, row.o2_y1, row.o2_x2, row.o2_y2, image_size)
            frame = frame[min(o1_y1, o2_y1):max(o1_y2, o2_y2), min(o1_x1, o2_x1):max(o1_x2, o2_x2)]
        # resize the frame to 224x224
        frame = cv2.resize(frame, (224, 224))
        return frame


    def expand_box(self,x1,y1,x2,y2,img_size,factor=1.5):
        H, W = img_size
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]

    def extract_features(self, frame):
        # image = Image.open(path).convert("RGB")
        inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        outputs = outputs.squeeze(0)
        return outputs

    def prepare_data(self):
        labeled_data = defaultdict(list) # dictionary with 'train' and 'test' fields. Each field is a list of tuples (image_features, label)
        # NOTE: it won't work well for relationships where the order of objects matters
        image_prompt = "{}? Answer with 'yes' or 'no'.".format(self.udf_description.rstrip(string.punctuation))
        logger.debug("Image prompt: {}".format(image_prompt))

        labeled_data_dir = os.path.join(self.config["model_dir"], "labeled_data", self.dataset)
        labeled_data_path = os.path.join(labeled_data_dir, "udf-{}_run-{}_ntrain-{}_labeled_data.pt".format(self.udf_class, self.run_id, self.n_train))
        os.makedirs(labeled_data_dir, exist_ok=True)
        if self.load_labeled_data and os.path.exists(labeled_data_path):
            labeled_data = torch.load(labeled_data_path)
            for data in labeled_data['train']:
                logger.debug("base64_image: {}".format(data["base64_image"]))
                logger.debug("gt_label: {}, llm_label: {}".format(data["label"], data["llm_label"]))
            logger.debug("llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}".format(labeled_data["metadata"]["llm_TP"], labeled_data["metadata"]["llm_FP"], labeled_data["metadata"]["llm_TN"], labeled_data["metadata"]["llm_FN"]))
            logger.debug("test_pos: {}, test_neg: {}".format(labeled_data["metadata"]["test_pos"], labeled_data["metadata"]["test_neg"]))
        else:
            llm_TP, llm_FP, llm_TN, llm_FN = 0, 0, 0, 0
            # Training and validation data
            label_count = 0
            for row in self.df_train.itertuples():
                try:
                    # Read and crop frame
                    print("row", row)
                    frame = self.frame_processing(row)
                    if frame is None:
                        continue
                    # Convert the frame to a base 64 encoded image
                    _, buffer = cv2.imencode('.jpg', frame)
                    base64_image = base64.b64encode(buffer).decode('utf-8')
                    logger.debug("base64_image: {}".format(base64_image))
                    # training_images.append(base64_image)
                    response = client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": image_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=10,
                        temperature=0.2,
                        top_p=0.5,
                        seed=self.run_id
                    )
                    result = response.choices[0].message.content
                    logger.debug("Result: {}".format(result))
                    gt_label = int(self.gt_udf(row.o1) if self.n_obj == 1 else self.gt_udf(row.o1, row.o2))
                    logger.debug("gt_label: {}".format(gt_label))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if "yes" in result.lower():
                        image_features = self.extract_features(frame)
                        labeled_data['train'].append({"image_features": image_features, "label": gt_label, "llm_label": 1, "base64_image": base64_image})
                        label_count += 1
                        if gt_label == 1:
                            llm_TP += 1
                        else:
                            llm_FP += 1
                    elif "no" in result.lower():
                        image_features = self.extract_features(frame)
                        labeled_data['train'].append({"image_features": image_features, "label": gt_label, "llm_label": 0, "base64_image": base64_image})
                        label_count += 1
                        if gt_label == 0:
                            llm_TN += 1
                        else:
                            llm_FN += 1
                    else:
                        raise ValueError("Invalid response", result)
                    if label_count >= self.n_train:
                        break
                except Exception as e:
                    logger.debug("Error: {}".format(e))
                    continue
            logger.debug("llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}".format(llm_TP, llm_FP, llm_TN, llm_FN))
            labeled_data["metadata"] = {"llm_TP": llm_TP, "llm_FP": llm_FP, "llm_TN": llm_TN, "llm_FN": llm_FN}

            # Test data
            for row in self.df_test.itertuples():
                try:
                    # Read and crop frame
                    frame = self.frame_processing(row)
                    if frame is None: # failed to read the frame
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_features = self.extract_features(frame)
                    label = int(self.gt_udf(row.o1) if self.n_obj == 1 else self.gt_udf(row.o1, row.o2))
                    labeled_data['test'].append({"image_features": image_features, "label": label})
                except Exception as e:
                    logger.debug("Error: {}".format(e))
                    continue
            pos_count = sum([1 for data in labeled_data['test'] if data["label"] == 1])
            neg_count = sum([1 for data in labeled_data['test'] if data["label"] == 0])
            logger.debug("test_pos: {}, test_neg: {}".format(pos_count, neg_count))
            labeled_data["metadata"]["test_pos"] = pos_count
            labeled_data["metadata"]["test_neg"] = neg_count
            # save labeled_data to a file
            if self.save_labeled_data:
                torch.save(labeled_data, labeled_data_path)

        # use 20% of the training data as validation data
        train_dataset = CustomImageDataset(labeled_data['train'], train=True)
        train_set_size = int(len(train_dataset) * 0.8)
        valid_set_size = len(train_dataset) - train_set_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(self.run_id))

        class_counts = [sum(data["llm_label"] == i for data in labeled_data['train']) for i in range(2)]
        self.class_weights = torch.tensor([1.0 / count for count in class_counts]).to(self.device)
        logger.debug("class_counts: {}, class_weights: {}".format(class_counts, self.class_weights))

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_dataset = CustomImageDataset(labeled_data['test'], train=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    def train(self):
        # Define the model
        self.dim_in = self.clip_model.config.projection_dim
        logger.debug("dim_in: {}".format(self.dim_in)) # should be 512 for clip-vit-base-patch32
        self.mlp_model = mlp.MLP(self.dim_in, 2, logger, self.class_weights) # binary classification
        self.checkpoint_root = os.path.join(self.config["model_dir"], "model_udf", self.dataset, self.udf_class)
        self.checkpoint_filename = "udf-{}_run-{}_ntrain-{}".format(self.udf_class, self.run_id, self.n_train)
        os.makedirs(self.checkpoint_root, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.checkpoint_root,
            filename=self.checkpoint_filename,
            monitor="val_loss",
            mode="min",
        )
        callbacks=[checkpoint_callback]
        log_dir=os.path.join(self.config["output_dir"], 'tensorboard')
        pl_logger = pl.loggers.TensorBoardLogger(log_dir, name="udf-{}_run-{}_ntrain-{}.log".format(self.udf_class, self.run_id, self.n_train), default_hp_metric=False)
        earlystopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
        callbacks.append(earlystopping_callback)
        self.trainer = pl.Trainer(
            # deterministic=self.deterministic,
            max_epochs=50,
            devices=1,
            accelerator="auto",
            enable_progress_bar=True,
            enable_checkpointing=True,
            enable_model_summary=False,
            logger=pl_logger,
            default_root_dir=self.checkpoint_root,
            callbacks=callbacks,
            # check_val_every_n_epoch=5,
            # log_every_n_steps=min(50, len(dataset)-1),
            log_every_n_steps=1
        )

        self.trainer.fit(
            self.mlp_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )

    def test(self):
        # Inference:
        # model = mlp.MLP.load_from_checkpoint(self.dim_in, 2)
        logger.debug("test with last model: ")
        self.trainer.test(self.mlp_model, dataloaders=self.test_loader)
        logger.debug("test with best model: ")
        self.trainer.test(ckpt_path="best", dataloaders=self.test_loader)

if __name__ == "__main__":
    # python model_udf.py --run_id 0 --dataset "clevrer" --udf_class "color_red" --n_train 100 --load_labeled_data
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )

    parser = argparse.ArgumentParser()
    # parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--udf_class", type=str, help="UDF class name we want to generate")
    parser.add_argument("--n_train", type=int, help="number of training samples")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data")
    parser.add_argument("--n_obj", type=int, help="number of objects in the UDF arguments")
    # parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in UDF")
    # parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    # parser.add_argument("--budget", type=int, help="labeling budget")
    # parser.add_argument("--ask_for_gt_udf", action="store_true", help="Ask for the gt_udf name interactively if enabled")
    # parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")

    args = parser.parse_args()
    # query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    udf_class = args.udf_class
    n_train = args.n_train
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data
    n_obj = args.n_obj
    # allow_kwargs_in_udf = args.allow_kwargs_in_udf
    # num_parameter_search = args.num_parameter_search
    # labeling_budget = args.budget
    # ask_for_gt_udf = args.ask_for_gt_udf
    # num_interpretations = args.num_interpretations

    random.seed(run_id)
    np.random.seed(run_id)

    # input_query_file = config[dataset]["input_query_file"]
    # input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    # gt_dsl = input_query["dsl"]
    # user_query = input_query["question"]
    # positive_videos = input_query["positive_videos"]
    # y_true = [1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"])]
    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(config["log_dir"], "model_udf", dataset)
    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(
        os.path.join(
            log_dir,
            "udf-{}_run-{}_ntrain-{}.log".format(udf_class, run_id, n_train),
        ),
        mode="w",
    )
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )
    name_map = {
        "color_brown": {"signature": "Color_Brown(o0)", "description": "Whether the color of o0 is brown."},
        "color_purple": {"signature": "Color_Purple(o0)", "description": "Whether the color of o0 is purple."},
        "color_cyan": {"signature": "Color_Cyan(o0)", "description": "Whether the color of o0 is cyan."},
        "shape_cylinder": {"signature": "Shape_Cylinder(o0)", "description": "Whether the shape of o0 is cylinder."},
        "shape_cube": {"signature": "Shape_Cube(o0)", "description": "Whether the shape of o0 is cube."},
        "shape_sphere": {"signature": "Shape_Sphere(o0)", "description": "Whether the shape of o0 is sphere."},
        "material_metal": {"signature": "Material_Metal(o0)", "description": "Whether the material of o0 is metal."},
        "material_rubber": {"signature": "Material_Rubber(o0)", "description": "Whether the material of o0 is rubber."},
        "near": {"signature": "Near(o0, o1)", "description": "Whether o0 is near o1."},
        "far": {"signature": "Far(o0, o1)", "description": "Whether o0 is far away from o1."},
        "rightof": {"signature": "RightOf(o0, o1)", "description": "Whether o0 is on the right of o1."},
        "behind": {"signature": "Behind(o0, o1)", "description": "Whether o0 is behind o1."},
        "location_right": {"signature": "Location_Right(o0)", "description": "Whether o0 is on the right of the frame."},
        "location_bottom": {"signature": "Location_Bottom(o0)", "description": "Whether o0 is at the bottom of the frame."},
    }
    udf_signature = name_map[udf_class]["signature"]
    udf_description = name_map[udf_class]["description"]
    gt_udf_name = "gt_{}.gt_0".format(udf_class)
    md = ModelDistiller(config, prompt_config, dataset, udf_signature, udf_description, gt_udf_name, run_id, n_train, save_labeled_data, load_labeled_data)
    md.prepare_data()
    md.train()
    md.test()