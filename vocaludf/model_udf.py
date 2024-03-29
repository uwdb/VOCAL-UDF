import argparse
import json
import logging
import os
import random
import yaml
import numpy as np
import cv2
from PIL import Image, ImageDraw
import duckdb
import base64
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel, LlavaNextForConditionalGeneration, LlavaNextProcessor
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
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import pandas as pd

client = OpenAI()

# logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


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
    llm_method = "gpt4v"
    mlp_method = "clip"
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
        # clip_model_name = "openai/clip-vit-base-patch32"
        clip_model_name = os.path.join(self.config['model_dir'], 'clip-vit-base-patch32')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        # self.clip_model.save_pretrained(os.path.join(self.config['model_dir'], 'clip-vit-base-patch32'))
        # self.clip_processor.save_pretrained(os.path.join(self.config['model_dir'], 'clip-vit-base-patch32'))

        self.dim_in = self.clip_model.config.projection_dim

    def frame_processing(self, row):
        vid = row['vid'] if self.n_obj == 1 else row['o1_vid']
        fid = row['fid'] if self.n_obj == 1 else row['o1_fid']
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
            return None, None
        cap.release()
        image_size = frame.shape[:2]
        if self.n_obj == 1:
            x1, y1, x2, y2 = self.expand_box(row['x1'], row['y1'], row['x2'], row['y2'], image_size)
            frame = frame[y1:y2, x1:x2]
        else:
            o1_x1, o1_y1, o1_x2, o1_y2 = self.expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
            o2_x1, o2_y1, o2_x2, o2_y2 = self.expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
            frame = frame[min(o1_y1, o2_y1):max(o1_y2, o2_y2), min(o1_x1, o2_x1):max(o1_x2, o2_x2)]
        # resize the frame to 224x224
        frame = cv2.resize(frame, (224, 224))
        return frame, image_size

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

    def llm_annotate_data(self):
        labeled_data = defaultdict(list) # dictionary with 'train' and 'test' fields. Each field is a list of tuples (image_features, label)

        labeled_data_dir = os.path.join(self.config["model_dir"], "labeled_data", self.dataset, self.llm_method)
        labeled_data_path = os.path.join(labeled_data_dir, "udf-{}_run-{}_ntrain-{}_labeled_data.pt".format(self.udf_class, self.run_id, self.n_train))
        os.makedirs(labeled_data_dir, exist_ok=True)
        if self.load_labeled_data and os.path.exists(labeled_data_path):
            logger.info("Loading labeled data from {}".format(labeled_data_path))
            labeled_data = torch.load(labeled_data_path)
            for data in labeled_data['train']:
                logger.debug("row: {}".format(data['row'].to_dict()))
                logger.debug("base64_image: {}".format(data["base64_image"]))
                logger.debug("gt_label: {}, llm_label: {}".format(data["label"], data["llm_label"]))
            logger.debug("llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(labeled_data["metadata"]["llm_TP"], labeled_data["metadata"]["llm_FP"], labeled_data["metadata"]["llm_TN"], labeled_data["metadata"]["llm_FN"], labeled_data["metadata"]["llm_f1"]))
            logger.debug("test_pos: {}, test_neg: {}".format(labeled_data["metadata"]["test_pos"], labeled_data["metadata"]["test_neg"]))
        else:
            self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN = 0, 0, 0, 0
            # Training and validation data
            self.label_count = 0
            for _, row in self.df_train.iterrows():
                try:
                    gt_label = self._get_gt_label(row)
                    logger.debug("gt_label: {}".format(gt_label))
                    # Read and crop frame
                    logger.debug("row: {}".format(row.to_dict()))
                    frame, image_size = self.frame_processing(row)
                    if frame is None:
                        continue
                    llm_label, base64_image, image_prompt = self._llm_annotate_frame(frame, image_size, row, gt_label)
                    labeled_data['train'].append({"label": gt_label, "llm_label": llm_label, "base64_image": base64_image, "image_prompt": image_prompt, "row": row})
                    if self.label_count >= self.n_train:
                        break
                except Exception as e:
                    logger.debug("Error: {}".format(e))
                    continue
            llm_f1 = 2*self.llm_TP/(2*self.llm_TP+self.llm_FP+self.llm_FN) if 2*self.llm_TP+self.llm_FP+self.llm_FN > 0 else 0.0
            logger.debug("llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN, llm_f1))
            labeled_data["metadata"] = {"llm_TP": self.llm_TP, "llm_FP": self.llm_FP, "llm_TN": self.llm_TN, "llm_FN": self.llm_FN, "llm_f1": llm_f1}

            # Test data
            for _, row in self.df_test.iterrows():
                try:
                    gt_label = self._get_gt_label(row)
                    labeled_data['test'].append({"label": gt_label, "row": row})
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
                logger.info("Saving labeled data to {}".format(labeled_data_path))
                torch.save(labeled_data, labeled_data_path)
        self.labeled_data = labeled_data

    def _llm_annotate_frame(self, frame, image_size, row, gt_label):
        # Convert the frame to a base 64 encoded image
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        logger.debug("base64_image: {}".format(base64_image))
        image_prompt = self._create_image_prompt(row, image_size)
        logger.debug("Image prompt: {}".format(image_prompt))
        response = completion_with_backoff(
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
        if "yes" in result.lower():
            llm_label = 1
            if gt_label == 1:
                self.llm_TP += 1
            else:
                self.llm_FP += 1
        elif "no" in result.lower():
            llm_label = 0
            if gt_label == 0:
                self.llm_TN += 1
            else:
                self.llm_FN += 1
        else:
            raise ValueError("Invalid response", result)
        self.label_count += 1
        return llm_label, base64_image, image_prompt

    def _create_image_prompt(self, row, image_size):
        # NOTE: it won't work well for relationships where the order of objects matters
        image_prompt = "{}? Answer with 'yes' or 'no'.".format(self.udf_description.rstrip(string.punctuation))
        return image_prompt

    def _get_gt_label(self, row):
        return int(self.gt_udf(row['o1']) if self.n_obj == 1 else self.gt_udf(row['o1'], row['o2']))

    def mlp_prepare_data(self):
        for split in ['train', 'test']:
            logger.info("Processing {} data".format(split))
            idx_to_remove = []
            for i in range(len(self.labeled_data[split])):
                try:
                    data = self.labeled_data[split][i]
                    row = data['row']
                    logger.debug("row: {}".format(row.to_dict()))
                    frame, image_size = self.frame_processing(row)
                    if frame is None: # failed to read the frame
                        idx_to_remove.append(i)
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_features = self.extract_features(frame, row, image_size)
                    self.labeled_data[split][i]["image_features"] = image_features
                except Exception as e:
                    logger.debug("Error: {}".format(e))
                    idx_to_remove.append(i)
                    continue
            for i in reversed(idx_to_remove):
                del self.labeled_data[split][i]

        # use 20% of the training data as validation data
        train_dataset = CustomImageDataset(self.labeled_data['train'], train=True)
        train_set_size = int(len(train_dataset) * 0.8)
        valid_set_size = len(train_dataset) - train_set_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(self.run_id))

        class_counts = [sum(data["llm_label"] == i for data in self.labeled_data['train']) for i in range(2)]
        self.class_weights = torch.tensor([1.0 / count for count in class_counts]).to(self.device)
        logger.debug("class_counts: {}, class_weights: {}".format(class_counts, self.class_weights))

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_dataset = CustomImageDataset(self.labeled_data['test'], train=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    def extract_features(self, frame, row, image_size):
        inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        outputs = outputs.squeeze(0)
        return outputs

    def train(self):
        # logger.debug("mlp_config: {}".format(mlp_config))
        logger.debug("dim_in: {}".format(self.dim_in)) # should be 512 for clip-vit-base-patch32
        self.checkpoint_root = os.path.join(self.config["model_dir"], "model_udf", self.dataset, f"{self.llm_method}_{self.mlp_method}", self.udf_class)
        self.checkpoint_filename = "udf-{}_run-{}_ntrain-{}".format(self.udf_class, self.run_id, self.n_train)
        os.makedirs(self.checkpoint_root, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=self.checkpoint_filename,
            monitor="val_loss",
            mode="min",
        )
        callbacks=[checkpoint_callback]
        log_dir=os.path.join(self.config["output_dir"], 'tensorboard', self.dataset, f"{self.llm_method}_{self.mlp_method}", self.udf_class)
        pl_logger = pl.loggers.TensorBoardLogger(log_dir, name="udf-{}_run-{}_ntrain-{}.log".format(self.udf_class, self.run_id, self.n_train), default_hp_metric=False)
        earlystopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
        callbacks.append(earlystopping_callback)

        self.mlp_model = mlp.MLP(self.dim_in, 2, logger, self.class_weights) # binary classification

        self.trainer = pl.Trainer(
            # deterministic=self.deterministic,
            max_epochs=50,
            devices=1,
            accelerator="auto",
            enable_progress_bar=True,
            enable_checkpointing=True,
            enable_model_summary=False,
            # logger=pl_logger,
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

class LlavaModelDistiller(ModelDistiller):
    llm_method = "llava_v1.6_mistral_7b"
    mlp_method = "clip"
    def __init__(self, config, prompt_config, dataset, udf_signature, udf_description, gt_udf_name, run_id, n_train, save_labeled_data, load_labeled_data):
        super().__init__(config, prompt_config, dataset, udf_signature, udf_description, gt_udf_name, run_id, n_train, save_labeled_data, load_labeled_data)

        # llava_model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        llava_model_name = os.path.join(self.config['model_dir'], 'llava-v1.6-mistral-7b-hf')
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            llava_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            # device_map="auto",
        # )
        ).to(self.device)
        self.llava_processor = LlavaNextProcessor.from_pretrained(llava_model_name)
        self.llava_processor.tokenizer.padding_side = "left"
        # self.llava_model.save_pretrained(os.path.join(self.config['model_dir'], 'llava-v1.6-mistral-7b-hf'))
        # self.llava_processor.save_pretrained(os.path.join(self.config['model_dir'], 'llava-v1.6-mistral-7b-hf'))

    def _llm_annotate_frame(self, frame, image_size, row, gt_label):
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        logger.debug("base64_image: {}".format(base64_image))
        # Convert the frame to PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        image_prompt, last_word = self._create_image_prompt(row, image_size)
        logger.debug("Image prompt: {}".format(image_prompt))
        inputs = self.llava_processor(image_prompt, pil_image, return_tensors="pt").to(self.device)
        output = self.llava_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.2,
            top_p=0.7
        )
        result = self.llava_processor.decode(output[0], skip_special_tokens=True)
        # logger.debug("Raw result: {}".format(result))
        result = result.split(last_word)[-1].strip().lower()
        logger.debug("Result: {}".format(result))
        if "yes" in result.lower():
            llm_label = 1
            if gt_label == 1:
                self.llm_TP += 1
            else:
                self.llm_FP += 1
        elif "no" in result.lower():
            llm_label = 0
            if gt_label == 0:
                self.llm_TN += 1
            else:
                self.llm_FN += 1
        else:
            raise ValueError("Invalid response", result)
        self.label_count += 1
        return llm_label, base64_image, image_prompt

    def _create_image_prompt(self, row, image_size):
        # NOTE: it won't work well for relationships where the order of objects matters
        image_prompt = "[INST] <image>\n{}? Answer with 'yes' or 'no'. [/INST]".format(self.udf_description.rstrip(string.punctuation))
        return image_prompt, "[/INST]"

class Llava34bModelDistiller(LlavaModelDistiller):
    llm_method = "llava-v1.6-34b-hf"
    mlp_method = "clip"
    def __init__(self, config, prompt_config, dataset, udf_signature, udf_description, gt_udf_name, run_id, n_train, save_labeled_data, load_labeled_data):
        super().__init__(config, prompt_config, dataset, udf_signature, udf_description, gt_udf_name, run_id, n_train, save_labeled_data, load_labeled_data)

        # llava_model_name = "llava-hf/llava-v1.6-34b-hf"
        llava_model_name = os.path.join(self.config['model_dir'], 'llava-v1.6-34b-hf')
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            llava_model_name,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.float16,
            # attn_implementation="flash_attention_2",
            device_map="auto",
        )
        # ).to(self.device)
        logger.debug("llava_model.hf_device_map: {}".format(self.llava_model.hf_device_map))
        self.llava_processor = LlavaNextProcessor.from_pretrained(llava_model_name)
        self.llava_processor.tokenizer.padding_side = "left"
        # self.llava_model.save_pretrained(os.path.join(self.config['model_dir'], 'llava-v1.6-34b-hf'))
        # self.llava_processor.save_pretrained(os.path.join(self.config['model_dir'], 'llava-v1.6-34b-hf'))

    def _create_image_prompt(self, row, image_size):
        # NOTE: it won't work well for relationships where the order of objects matters
        image_prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{}? Answer with 'yes' or 'no'.<|im_end|><|im_start|>assistant\n".format(self.udf_description.rstrip(string.punctuation))
        return image_prompt, "assistant"

class BoundingBoxAnnotatedModelDistiller(ModelDistiller):
    """
    Annotate image with bounding boxes (subject: red box, target: blue box) so that the model is aware of the subject and the target in the image.
    The subject is annotated with a red bounding box, and the target is annotated with a blue bounding box
    """
    llm_method = "gpt4v_norm_bbox"
    mlp_method = "clip_norm_bbox"
    def __init__(self, config, prompt_config, dataset, udf_signature, udf_description, gt_udf_name, run_id, n_train, save_labeled_data, load_labeled_data):
        super().__init__(config, prompt_config, dataset, udf_signature, udf_description, gt_udf_name, run_id, n_train, save_labeled_data, load_labeled_data)
        if self.n_obj == 2:
            self.dim_in += 8 # concat with bounding box features

    def _create_image_prompt(self, row, image_size):
        if self.n_obj == 1:
            image_prompt = "{}? Answer with 'yes' or 'no'.".format(self.udf_description.rstrip(string.punctuation))
        else:
            boxes = [(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2']), (row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'])]
            # normalize the bounding box coordinates by the image size
            boxes = [(round(x1/image_size[1], 3), round(y1/image_size[0], 3), round(x2/image_size[1], 3), round(y2/image_size[0], 3)) for x1, y1, x2, y2 in boxes]

            image_prompt = "{}? Answer with 'yes' or 'no'.".format(self.replace_objects(self.udf_description.rstrip(string.punctuation), boxes))
        return image_prompt

    def replace_objects(self, input_string, boxes):
        # NOTE: self.n_obj must be 2
        assert len(boxes) == 2, "boxes must have 2 elements"
        # Find all occurrences of "o" followed by integers
        objects = re.findall(r'o\d+', input_string)
        # Sort the objects based on the integer part of the identifier
        sorted_objects = sorted(objects, key=lambda x: int(x[1:]))

        if len(sorted_objects) == 2:
            new_string = input_string.replace(sorted_objects[0], f"{sorted_objects[0]} at {boxes[0]}")
            new_string = new_string.replace(sorted_objects[1], f"{sorted_objects[1]} at {boxes[1]}")

        return new_string

    def extract_features(self, frame, row, image_size):
        inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        outputs = outputs.squeeze(0)
        if self.n_obj == 2:
            h, w = image_size
            box_features = torch.tensor([row['o1_x1']*1.0/w, row['o1_y1']*1.0/h, row['o1_x2']*1.0/w, row['o1_y2']*1.0/h, row['o2_x1']*1.0/w, row['o2_y1']*1.0/h, row['o2_x2']*1.0/w, row['o2_y2']*1.0/h]).to(self.device)
            outputs = torch.cat((outputs, box_features))
        return outputs

class GQARelationshipModelDistiller(ModelDistiller):
    llm_method = "gpt4v"
    mlp_method = "clip"
    def __init__(self, config, prompt_config, dataset, udf_signature, udf_description, run_id, n_train, save_labeled_data, load_labeled_data):
        self.config = config
        self.prompt_config = prompt_config
        self.dataset = dataset
        udf_name, udf_vars = parse_signature(udf_signature)
        self.udf_class = udf_name.lower()
        self.n_obj = len(udf_vars)
        assert self.n_obj == 2, "n_obj must be 2"
        self.udf_description = udf_description
        self.run_id = run_id
        self.n_train = n_train
        self.n_test = 1000
        self.save_labeled_data = save_labeled_data
        self.load_labeled_data = load_labeled_data

        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )

        # NOTE: LLM doesn't generate labels in some cases, so we need to double the number of samples (i.e., self.n_train * 2) to ensure we have enough training samples
        self.df_train, self.df_test = self.construct_train_and_test_data(self.n_obj, self.n_train * 2, self.n_test)

        # Load the CLIP model
        model_name = os.path.join(self.config['model_dir'], 'clip-vit-base-patch32')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        self.dim_in = self.clip_model.config.projection_dim

    def construct_train_and_test_data(self, n_obj, n_train, n_test=None):
        # Construct training data and test data
        self.conn.execute(f"SELECT setseed({self.run_id / 100})")
        if n_obj == 1:
            raise ValueError("Number of objects not supported: {}".format(n_obj))
        elif n_obj == 2:
            schema = self.conn.execute("DESCRIBE gqa_objects").df()
            names = schema["column_name"].values
            # o1.fid as o1_fid, o1.oid as o1_oid
            project_clause = (
                "m.width as width, m.height as height, " +
                ", ".join(["o1.{} as o1_{}".format(name, name) for name in names])
                + ", "
                + ", ".join(["o2.{} as o2_{}".format(name, name) for name in names])
            )
            df_filtered = self.conn.execute(
                """
                SELECT {}
                FROM gqa_metadata m, gqa_objects o1, gqa_objects o2
                WHERE m.fid = o1.fid AND o1.fid = o2.fid AND o1.oid != o2.oid
                ORDER BY random()
                LIMIT {}
            """.format(
                    project_clause, n_train + n_test if n_test else n_train
                )
            ).df()
            df_filtered["o1"] = df_filtered.apply(
                lambda row: {
                    col.split("_", 1)[1]: row[col]
                    for col in df_filtered.columns
                    if col.startswith("o1_")
                },
                axis=1,
            )
            df_filtered["o2"] = df_filtered.apply(
                lambda row: {
                    col.split("_", 1)[1]: row[col]
                    for col in df_filtered.columns
                    if col.startswith("o2_")
                },
                axis=1,
            )
        else:
            raise ValueError("Number of objects not supported: {}".format(n_obj))
        if n_test:
            df_train = df_filtered[:n_train]
            df_train = df_train.reset_index()
            df_test = df_filtered[n_train:]
            df_test = df_test.reset_index()
            return df_train, df_test
        else:
            df_filtered = df_filtered.reset_index()
            return df_filtered

    def frame_processing(self, row):
        fid = row['fid'] if self.n_obj == 1 else row['o1_fid']
        frame = cv2.imread(os.path.join(
                self.config['data_dir'],
                self.dataset,
                "images",
                str(fid % 10),
                f"{fid}.jpg"
            ))
        image_size = frame.shape[:2]
        if self.n_obj == 1:
            raise ValueError("Number of objects not supported: {}".format(self.n_obj))
        else:
            o1_x1, o1_y1, o1_x2, o1_y2 = self.expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
            o2_x1, o2_y1, o2_x2, o2_y2 = self.expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
            frame = frame[min(o1_y1, o2_y1):max(o1_y2, o2_y2), min(o1_x1, o2_x1):max(o1_x2, o2_x2)]
        # resize the frame to 224x224
        frame = cv2.resize(frame, (224, 224))
        return frame, image_size

    def _get_gt_label(self, row):
        df = self.conn.execute(
            """
            SELECT *
            FROM gqa_relationships
            WHERE fid = {} AND oid1 = {} AND oid2 = {} AND rname = '{}'
            """.format(
                row['o1_fid'], row['o1_oid'], row['o2_oid'], self.udf_class.replace("_", " ")
            )
        ).df()
        # logger.debug("df: {}".format(df))
        return 1 if len(df) > 0 else 0

    def _create_image_prompt(self, row, image_size):
        image_prompt = "{}? Answer with 'yes' or 'no'.".format(self.replace_objects(self.udf_description.rstrip(string.punctuation), row, image_size))
        return image_prompt

    def replace_objects(self, input_string, row, image_size):
        # Find all occurrences of "o" followed by integers
        objects = re.findall(r'o\d+', input_string)
        # Sort the objects based on the integer part of the identifier
        sorted_objects = sorted(objects, key=lambda x: int(x[1:]))

        h, w = image_size
        if len(sorted_objects) == 2:
            new_string = input_string.replace(sorted_objects[0], f"{row['o1_oname']} {sorted_objects[0]} at {round(row['o1_x1']/w, 3), round(row['o1_y1']/h, 3), round(row['o1_x2']/w, 3), round(row['o1_y2']/h, 3)}")
            new_string = new_string.replace(sorted_objects[1], f"{row['o2_oname']} {sorted_objects[1]} at {round(row['o2_x1']/w, 3), round(row['o2_y1']/h, 3), round(row['o2_x2']/w, 3), round(row['o2_y2']/h, 3)}")

        return new_string

class GQARelationshipLlavaModelDistiller(LlavaModelDistiller, GQARelationshipModelDistiller):
    llm_method = "llava_v1.6_mistral_7b"
    mlp_method = "clip"
    def __init__(self, config, prompt_config, dataset, udf_signature, udf_description, run_id, n_train, save_labeled_data, load_labeled_data):
        GQARelationshipModelDistiller.__init__(self, config, prompt_config, dataset, udf_signature, udf_description, run_id, n_train, save_labeled_data, load_labeled_data)

        llava_model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        # llava_model_name = os.path.join(self.config['model_dir'], 'llava-v1.6-mistral-7b-hf')
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            llava_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            # device_map="auto",
        # )
        ).to(self.device)
        self.llava_processor = LlavaNextProcessor.from_pretrained(llava_model_name)
        self.llava_processor.tokenizer.padding_side = "left"
        self.llava_model.save_pretrained(os.path.join(self.config['model_dir'], 'llava-v1.6-mistral-7b-hf'))
        self.llava_processor.save_pretrained(os.path.join(self.config['model_dir'], 'llava-v1.6-mistral-7b-hf'))

    def _create_image_prompt(self, row, image_size):
        image_prompt = "[INST] <image>\n{}? Answer with 'yes' or 'no'. [/INST]".format(self.replace_objects(self.udf_description.rstrip(string.punctuation), row, image_size))
        return image_prompt, "[/INST]"

class GQARelationshipLlava34bModelDistiller(GQARelationshipLlavaModelDistiller):
    llm_method = "llava-v1.6-34b-hf"
    mlp_method = "clip"
    def __init__(self, config, prompt_config, dataset, udf_signature, udf_description, run_id, n_train, save_labeled_data, load_labeled_data):
        GQARelationshipModelDistiller.__init__(self, config, prompt_config, dataset, udf_signature, udf_description, run_id, n_train, save_labeled_data, load_labeled_data)

        llava_model_name = "llava-hf/llava-v1.6-34b-hf"
        # llava_model_name = os.path.join(self.config['model_dir'], 'llava-v1.6-34b-hf')
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            llava_model_name,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.float16,
            # attn_implementation="flash_attention_2",
            device_map="auto",
        )
        # ).to(self.device)
        logger.debug("llava_model.hf_device_map: {}".format(self.llava_model.hf_device_map))
        self.llava_processor = LlavaNextProcessor.from_pretrained(llava_model_name)
        self.llava_processor.tokenizer.padding_side = "left"
        self.llava_model.save_pretrained(os.path.join(self.config['model_dir'], 'llava-v1.6-34b-hf'))
        self.llava_processor.save_pretrained(os.path.join(self.config['model_dir'], 'llava-v1.6-34b-hf'))

    def _create_image_prompt(self, row, image_size):
        image_prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{}? Answer with 'yes' or 'no'.<|im_end|><|im_start|>assistant\n".format(self.replace_objects(self.udf_description.rstrip(string.punctuation), row, image_size))
        return image_prompt, "assistant"

class GQARelationshipBalancedModelDistiller(GQARelationshipModelDistiller):
    llm_method = "balanced_gpt4v"
    mlp_method = "balanced_clip"
    """
    Instead of randomly sampling training data, we sample the same number of positive and negative examples,
    to see whether LLM can label the class correctly and whether the model can learn the class.
    """
    def construct_train_and_test_data(self, n_obj, n_train, n_test=None):
        # Construct training data and test data
        self.conn.execute(f"SELECT setseed({self.run_id / 100})")
        if n_obj == 1:
            raise ValueError("Number of objects not supported: {}".format(n_obj))
        elif n_obj == 2:
            schema = self.conn.execute("DESCRIBE gqa_objects").df()
            names = schema["column_name"].values
            # o1.fid as o1_fid, o1.oid as o1_oid
            project_clause = (
                "m.width as width, m.height as height, " +
                ", ".join(["o1.{} as o1_{}".format(name, name) for name in names])
                + ", "
                + ", ".join(["o2.{} as o2_{}".format(name, name) for name in names])
            )
            pos_sql = """
                SELECT {}, 1 as label
                FROM gqa_metadata m, gqa_objects o1, gqa_objects o2, gqa_relationships r
                WHERE m.fid = o1.fid AND o1.fid = o2.fid AND o1.oid != o2.oid
                    AND o1.fid = r.fid AND o1.oid = r.oid1 AND o2.oid = r.oid2 AND r.rname = '{}'
                ORDER BY random()
                LIMIT {}
            """.format(
                project_clause, self.udf_class.replace("_", " "), (n_train + n_test) // 2 if n_test else (n_train) // 2
            )
            neg_sql = """
                SELECT {}, 0 as label
                FROM gqa_metadata m, gqa_objects o1, gqa_objects o2
                WHERE m.fid = o1.fid AND o1.fid = o2.fid AND o1.oid != o2.oid
                    AND NOT EXISTS (
                        SELECT 1
                        FROM gqa_relationships r
                        WHERE r.fid = o1.fid AND r.oid1 = o1.oid
                            AND r.oid2 = o2.oid AND r.rname = '{}'
                    )
                ORDER BY random()
                LIMIT {}
            """.format(
                project_clause, self.udf_class.replace("_", " "), (n_train + n_test) // 2 if n_test else (n_train) // 2
            )
            # logger.debug("pos_sql: {}".format(pos_sql))
            # logger.debug("neg_sql: {}".format(neg_sql))
            df_pos = self.conn.execute(pos_sql).df()
            df_neg = self.conn.execute(neg_sql).df()
            for df in [df_pos, df_neg]:
                df["o1"] = df.apply(
                    lambda row: {
                        col.split("_", 1)[1]: row[col]
                        for col in df.columns
                        if col.startswith("o1_")
                    },
                    axis=1,
                )
                df["o2"] = df.apply(
                    lambda row: {
                        col.split("_", 1)[1]: row[col]
                        for col in df.columns
                        if col.startswith("o2_")
                    },
                    axis=1,
                )
            # Interleave the rows of the two DataFrames
            df_filtered = pd.concat([df_pos, df_neg], ignore_index=True)
            max_length = max(len(df_pos), len(df_neg))
            new_index = np.array([[i, i + max_length] for i in range(max_length)]).flatten()
            new_index = new_index[new_index < len(df_filtered)]
            df_filtered = df_filtered.iloc[new_index]
        else:
            raise ValueError("Number of objects not supported: {}".format(n_obj))
        if n_test:
            df_train = df_filtered[:n_train]
            df_train = df_train.reset_index()
            df_test = df_filtered[n_train:]
            df_test = df_test.reset_index()
            return df_train, df_test
        else:
            df_filtered = df_filtered.reset_index()
            return df_filtered

    def _get_gt_label(self, row):
        return row['label']


class GQARelationshipUnnormBboxBalancedModelDistiller(GQARelationshipBalancedModelDistiller):
    llm_method = "balanced_gpt4v"
    mlp_method = "balanced_clip_unnorm_bbox"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.load_labeled_data
        self.dim_in += 8

    def extract_features(self, frame, row, image_size):
        """
        CLIP features + un-normalized bbox coordinates for each object
        """
        inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        outputs = outputs.squeeze(0)
        if self.n_obj == 2:
            box_features = torch.tensor([row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2']], dtype=torch.float32).to(self.device)
            outputs = torch.cat((outputs, box_features))
        return outputs

class GQARelationshipLlavaUnnormBboxBalancedModelDistiller(GQARelationshipLlavaModelDistiller, GQARelationshipUnnormBboxBalancedModelDistiller):
    llm_method = "balanced_llava_v1.6_mistral_7b"
    mlp_method = "balanced_clip_unnorm_bbox"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_in += 8

class GQARelationshipNormBboxBalancedModelDistiller(GQARelationshipBalancedModelDistiller):
    llm_method = "balanced_gpt4v"
    mlp_method = "balanced_clip_norm_bbox"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.load_labeled_data
        self.dim_in += 8

    def extract_features(self, frame, row, image_size):
        """
        CLIP features + normalized bbox coordinates (x1/w, y1/h, x2/w, y2/h) for each object
        """
        inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        outputs = outputs.squeeze(0)
        if self.n_obj == 2:
            h, w = image_size
            box_features = torch.tensor([row['o1_x1']*1.0/w, row['o1_y1']*1.0/h, row['o1_x2']*1.0/w, row['o1_y2']*1.0/h, row['o2_x1']*1.0/w, row['o2_y1']*1.0/h, row['o2_x2']*1.0/w, row['o2_y2']*1.0/h]).to(self.device)
            outputs = torch.cat((outputs, box_features))
        return outputs

class GQARelationshipLlavaNormBboxBalancedModelDistiller(GQARelationshipLlavaModelDistiller, GQARelationshipNormBboxBalancedModelDistiller):
    llm_method = "balanced_llava_v1.6_mistral_7b"
    mlp_method = "balanced_clip_norm_bbox"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_in += 8

class GQARelationshipTwoCLIPBalancedModelDistiller(GQARelationshipBalancedModelDistiller):
    llm_method = "balanced_gpt4v"
    mlp_method = "balanced_two_clip"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.load_labeled_data
        self.dim_in *= 2

    def extract_features(self, frame, row, image_size):
        """
        CLIP features of image with subject in red box + CLIP features of image with target in red box
        """
        if self.n_obj == 2:
            o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_subject = frame.copy()
            frame_target = frame.copy()
            # draw bounding boxe of subject on the image
            cv2.rectangle(frame_subject, (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
            cv2.rectangle(frame_target, (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(0, 0, 255), thickness=1)
            _, buffer = cv2.imencode('.jpg', frame_subject)
            base64_frame_subject = base64.b64encode(buffer).decode('utf-8')
            logger.debug("base64_frame_subject: {}".format(base64_frame_subject))
            _, buffer = cv2.imencode('.jpg', frame_target)
            base64_frame_target = base64.b64encode(buffer).decode('utf-8')
            logger.debug("base64_frame_target: {}".format(base64_frame_target))
            frame_subject = cv2.cvtColor(frame_subject, cv2.COLOR_BGR2RGB)
            frame_target = cv2.cvtColor(frame_target, cv2.COLOR_BGR2RGB)
            inputs = self.clip_processor(images=[frame_subject, frame_target], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            outputs = outputs.reshape(-1)
        return outputs

    def _compute_new_box_after_crop(self, row, image_size):
        o1_x1, o1_y1, o1_x2, o1_y2 = self.expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
        o2_x1, o2_y1, o2_x2, o2_y2 = self.expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
        x_offset = min(o1_x1, o2_x1)
        y_offset = min(o1_y1, o2_y1)
        h_ratio = 224.0 / (max(o1_y2, o2_y2) - y_offset)
        w_ratio = 224.0 / (max(o1_x2, o2_x2) - x_offset)
        return (row['o1_x1'] - x_offset) * w_ratio, (row['o1_y1'] - y_offset) * h_ratio, (row['o1_x2'] - x_offset) * w_ratio, (row['o1_y2'] - y_offset) * h_ratio, (row['o2_x1'] - x_offset) * w_ratio, (row['o2_y1'] - y_offset) * h_ratio, (row['o2_x2'] - x_offset) * w_ratio, (row['o2_y2'] - y_offset) * h_ratio


class GQARelationshipLlavaTwoCLIPBalancedModelDistiller(GQARelationshipLlavaModelDistiller, GQARelationshipTwoCLIPBalancedModelDistiller):
    llm_method = "balanced_llava_v1.6_mistral_7b"
    mlp_method = "balanced_two_clip"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_in *= 2


class GQARelationshipThreeCLIPBalancedModelDistiller(GQARelationshipBalancedModelDistiller):
    llm_method = "balanced_gpt4v"
    mlp_method = "balanced_three_clip"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.load_labeled_data
        self.dim_in *= 3

    def extract_features(self, frame, row, image_size):
        """
        three CLIP features: original image, subject mask, target mask
        """
        if self.n_obj == 2:
            o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_subject = frame.copy()
            frame_target = frame.copy()
            # set the pixels outside the bounding box to 0
            frame_subject[:int(o1y1), :] = 0
            frame_subject[int(o1y2):, :] = 0
            frame_subject[:, :int(o1x1)] = 0
            frame_subject[:, int(o1x2):] = 0
            frame_target[:int(o2y1), :] = 0
            frame_target[int(o2y2):, :] = 0
            frame_target[:, :int(o2x1)] = 0
            frame_target[:, int(o2x2):] = 0
            _, buffer = cv2.imencode('.jpg', frame_subject)
            base64_frame_subject = base64.b64encode(buffer).decode('utf-8')
            logger.debug("base64_frame_subject: {}".format(base64_frame_subject))
            _, buffer = cv2.imencode('.jpg', frame_target)
            base64_frame_target = base64.b64encode(buffer).decode('utf-8')
            logger.debug("base64_frame_target: {}".format(base64_frame_target))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_subject = cv2.cvtColor(frame_subject, cv2.COLOR_BGR2RGB)
            frame_target = cv2.cvtColor(frame_target, cv2.COLOR_BGR2RGB)
            inputs = self.clip_processor(images=[frame, frame_subject, frame_target], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            outputs = outputs.reshape(-1)
        return outputs

    def _compute_new_box_after_crop(self, row, image_size):
        o1_x1, o1_y1, o1_x2, o1_y2 = self.expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
        o2_x1, o2_y1, o2_x2, o2_y2 = self.expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
        x_offset = min(o1_x1, o2_x1)
        y_offset = min(o1_y1, o2_y1)
        h_ratio = 224.0 / (max(o1_y2, o2_y2) - y_offset)
        w_ratio = 224.0 / (max(o1_x2, o2_x2) - x_offset)
        return (row['o1_x1'] - x_offset) * w_ratio, (row['o1_y1'] - y_offset) * h_ratio, (row['o1_x2'] - x_offset) * w_ratio, (row['o1_y2'] - y_offset) * h_ratio, (row['o2_x1'] - x_offset) * w_ratio, (row['o2_y1'] - y_offset) * h_ratio, (row['o2_x2'] - x_offset) * w_ratio, (row['o2_y2'] - y_offset) * h_ratio


class GQARelationshipLlavaThreeCLIPBalancedModelDistiller(GQARelationshipLlavaModelDistiller, GQARelationshipThreeCLIPBalancedModelDistiller):
    llm_method = "balanced_llava_v1.6_mistral_7b"
    mlp_method = "balanced_three_clip"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_in *= 3

class GQARelationshipLlava34bThreeCLIPBalancedModelDistiller(GQARelationshipLlava34bModelDistiller, GQARelationshipThreeCLIPBalancedModelDistiller):
    llm_method = "balanced_llava-v1.6-34b-hf"
    mlp_method = "balanced_three_clip"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_in *= 3


class GQARelationshipNormBboxOnlyBalancedModelDistiller(GQARelationshipBalancedModelDistiller):
    llm_method = "balanced_gpt4v"
    mlp_method = "balanced_norm_bbox_only"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.load_labeled_data
        self.dim_in = 8

    def extract_features(self, frame, row, image_size):
        if self.n_obj == 2:
            h, w = image_size
            box_features = torch.tensor([row['o1_x1']*1.0/w, row['o1_y1']*1.0/h, row['o1_x2']*1.0/w, row['o1_y2']*1.0/h, row['o2_x1']*1.0/w, row['o2_y1']*1.0/h, row['o2_x2']*1.0/w, row['o2_y2']*1.0/h]).to(self.device)
        return box_features


class GQARelationshipLlavaNormBboxOnlyBalancedModelDistiller(GQARelationshipLlavaModelDistiller, GQARelationshipNormBboxOnlyBalancedModelDistiller):
    llm_method = "balanced_llava_v1.6_mistral_7b"
    mlp_method = "balanced_norm_bbox_only"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_in = 8


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
