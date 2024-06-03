import argparse
import os
from vocaludf.utils import (
    replace_slot,
    parse_signature,
)
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from PIL import Image
import time
import resource
import duckdb
import logging
from openai import OpenAI
import re
from collections import defaultdict
import importlib
import numpy as np
from sklearn.metrics import f1_score
import copy
import cv2
from tqdm import tqdm
import sys
from transformers import CLIPProcessor, CLIPModel, LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch
from torch.utils.data import Dataset
import base64
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import string
import lightning.pytorch as pl
from vocaludf import mlp
import torchvision.ops as ops
import torchvision.transforms as T
import yaml
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook, get_active_domain

tqdm.pandas()
client = OpenAI()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

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


class UDFProposer:
    llm_method = "gpt4v"
    mlp_method = "three_clip"
    # Propose new UDFs and generate semantic interpretations
    def __init__(
        self,
        config,
        prompt_config,
        object_domain,
        relationship_domain,
        attribute_domain,
        dataset,
        query_id,
        run_id,
        num_workers,
        save_labeled_data,
        load_labeled_data,
        n_train_distill,
    ):
        self.config = config
        self.prompt_config = prompt_config
        self.object_domain = object_domain
        self.relationship_domain = relationship_domain
        self.attribute_domain = attribute_domain
        self.dataset = dataset
        self.query_id = query_id
        self.run_id = run_id
        self.num_workers = num_workers

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

        self.init_table()

        # Create a train and test split
        # NOTE: probably put these values in the config file
        self.n_train = 10000
        self.n_test = 10000

        # Initialization for model distillation
        self.n_train_distill = n_train_distill
        self.n_test_distill = 1000
        self.save_labeled_data = save_labeled_data
        self.load_labeled_data = load_labeled_data
        self.attribute_features_dir = os.path.join(self.config[self.dataset]["features_dir"], "attribute")
        self.relationship_features_dir = os.path.join(self.config[self.dataset]["features_dir"], "relationship")
        # Load the CLIP model
        # clip_model_name = "openai/clip-vit-base-patch32"
        clip_model_name = os.path.join(self.config['model_dir'], 'clip-vit-base-patch32')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.dim_in = self.clip_model.config.projection_dim

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


        if self.dataset == "charades":
            df_metadata = self.conn.execute(f"""
                SELECT DISTINCT vname, vid
                FROM charades_metadata
            """).df()
            self.vid_to_vname = {int(vid): vname for vid, vname in zip(df_metadata['vid'], df_metadata['vname'])}

    def init_table(self):
        metadata_join_clause = '' if self.dataset in ['clevr', 'clevrer'] else f'LEFT OUTER JOIN {self.dataset}_metadata m ON o1.vid = m.vid AND o1.fid = m.fid'
        height_width_clause = '320 AS height, 480 AS width' if self.dataset in ['clevr', 'clevrer'] else 'm.height AS height, m.width AS width'
        group_by_clause = 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2' if self.dataset in ['clevr', 'clevrer'] else 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2, m.height, m.width'
        # TODO: use object_domain to filter out objects that are not in the domain
        attr_parameters = ','.join('?' for _ in self.attribute_domain)
        sql = f"""
            SELECT
                o1.vid AS vid, o1.fid AS fid, o1.oid AS o1_oid, o1.oname AS o1_oname,
                o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
                {height_width_clause}
            FROM {self.dataset}_objects o1
            LEFT OUTER JOIN {self.dataset}_attributes a ON o1.vid = a.vid AND o1.fid = a.fid AND o1.oid = a.oid
            {metadata_join_clause}
            GROUP BY {group_by_clause}
            ORDER BY o1.vid, o1.fid, o1.oid, o1.x1, o1.y1, o1.x2, o1.y2
        """
        logger.debug(f"Create one_object table:\n{sql}")
        self.one_object_df = self.conn.execute(sql, self.attribute_domain).df()

        rel_parameters = ','.join('?' for _ in self.relationship_domain)
        sql = f"""
            WITH obj_with_attrs AS (
                SELECT
                    o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                    COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS attributes
                FROM {self.dataset}_objects o
                LEFT OUTER JOIN {self.dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid AND a.aname = ANY([{attr_parameters}])
                GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
            )
            , relationships_expanded AS (
                SELECT
                    vid, fid, oid1, oid2,
                    COALESCE(ARRAY_AGG(rname) FILTER (WHERE rname = ANY([{rel_parameters}])), ARRAY[]::varchar[]) AS rnames,
                    ARRAY_AGG(rname) AS gt_rnames
                FROM {self.dataset}_relationships
                GROUP BY vid, fid, oid1, oid2
            )
            SELECT
                o1.vid AS vid, o1.fid AS fid,
                o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2, o1.attributes AS o1_anames,
                o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2, o2.attributes AS o2_anames,
                COALESCE(r1.rnames, ARRAY[]::varchar[]) AS o1_o2_rnames,
                COALESCE(r2.rnames, ARRAY[]::varchar[]) AS o2_o1_rnames,
                COALESCE(r1.gt_rnames, ARRAY[]::varchar[]) AS o1_o2_gt_rnames,
                {height_width_clause}
            FROM obj_with_attrs o1
            JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid
            LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
            LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
            {metadata_join_clause}
            WHERE o1.oid <> o2.oid
            ORDER BY o1.vid, o1.fid, o1.oid, o2.oid, o1.x1, o1.y1, o1.x2, o1.y2, o2.x1, o2.y1, o2.x2, o2.y2
        """
        logger.debug(f"Create two_objects table:\n{sql}")
        self.two_objects_df = self.conn.execute(sql, self.attribute_domain + self.relationship_domain).df()

    def _distill_model(self, udf_signature, udf_description, gt_udf_name=None):
        """
        gt_udf_name: ground truth UDF name. If provided, compute llm's TP, FP, TN, FN, f1 score and test the trained model.
        """
        # Initialization for model distillation
        udf_name, udf_vars = parse_signature(udf_signature)
        self.udf_class = udf_name.lower()
        self.n_obj = len(udf_vars)
        assert self.n_obj in [1, 2], "n_obj must be 1 or 2"
        self.udf_description = udf_description

        self.gt_udf_name = gt_udf_name

        # TODO: ask LLM about relevant object classes to the target relationships, and filter data
        # filtered_objects = ['laptop', 'television', 'person', 'phone/camera']
        filtered_objects = None
        if self.dataset in ["charades"]:
            filtered_objects = list(set(self.llm_filter_relevant_objects(udf_signature, udf_description) + ['person']))
        logger.debug(f"filtered_objects: {filtered_objects}")

        # NOTE: LLM doesn't generate labels in some cases, so we need to double the number of samples (i.e., self.n_train * 2) to ensure we have enough training samples
        if gt_udf_name:
            self.df_train, self.df_test = self.construct_train_and_test_data(self.n_obj, self.n_train_distill * 2, self.n_test_distill, df_with_img_column=True, filtered_objects=filtered_objects)
        else:
            self.df_train = self.construct_train_and_test_data(self.n_obj, self.n_train_distill * 2, df_with_img_column=True, filtered_objects=filtered_objects)

        self.llm_annotate_data()
        # self.gt_annotate_data()
        self.mlp_prepare_data()
        best_ckpt = self.train()
        if gt_udf_name:
            self.test()

        udf_dict = {}
        udf_dict["udf_name"] = udf_name
        udf_dict["udf_signature"] = udf_signature
        udf_dict["udf_description"] = udf_description
        udf_dict["semantic_interpretation"] = 'model'
        udf_dict["function_implementation"] = best_ckpt

    def llm_filter_relevant_objects(self, udf_signature, udf_description):
        filter_objects_prompt = replace_slot(
            self.prompt_config["filter_training_data"],
            {
                "object_classes": self.object_domain,
                "udf_signature": udf_signature,
                "udf_description": udf_description,
            },
        )
        logger.debug("filter_objects_prompt: {}".format(filter_objects_prompt))
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                logger.debug(f"trial: {trial}")
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-2024-04-09",
                    messages=[
                        {"role": "user", "content": filter_objects_prompt},
                    ],
                    temperature=self.config["udf_generator"]["temperature"],
                    top_p=self.config["udf_generator"]["top_p"],
                    seed=self.run_id * 42 + trial,
                )
                filtered_objects = eval(
                    "\n\n".join(
                        re.findall(
                            r"```json\n(.*?)```",
                            response.choices[0].message.content,
                            re.DOTALL,
                        )
                    )
                )["answer"]
                return filtered_objects
            except Exception as e:
                logger.exception("ERROR: failed to filter relevant objects: {}".format(e))
                logger.debug(response)

    def gt_annotate_data(self):
        labeled_data = defaultdict(list)
        self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN = 0, 0, 0, 0
        # Training and validation data
        self.label_count = 0
        for _, row in self.df_train.iterrows():
            logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++")
            try:
                gt_label = self._get_gt_label(row)
                # Read and crop frame
                logger.debug("row: {}".format(row.drop('img').to_dict()))
                labeled_data['train'].append({"label": gt_label, "llm_label": gt_label, "base64_image": None, "image_prompt": "", "row": row})
                self.label_count += 1
                if self.label_count >= self.n_train_distill:
                    break
            except Exception as e:
                logger.exception("Error: {}".format(e))
                continue
        if self.gt_udf_name is not None:
            llm_f1 = 2*self.llm_TP/(2*self.llm_TP+self.llm_FP+self.llm_FN) if 2*self.llm_TP+self.llm_FP+self.llm_FN > 0 else 0.0
            logger.debug("llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN, llm_f1))
        else:
            llm_f1 = -1
        labeled_data["metadata"] = {"llm_TP": self.llm_TP, "llm_FP": self.llm_FP, "llm_TN": self.llm_TN, "llm_FN": self.llm_FN, "llm_f1": llm_f1}

        # Test data
        if self.gt_udf_name is not None:
            for _, row in self.df_test.iterrows():
                try:
                    gt_label = self._get_gt_label(row)
                    labeled_data['test'].append({"label": gt_label, "row": row})
                except Exception as e:
                    logger.exception("Error: {}".format(e))
                    continue
            pos_count = sum([1 for data in labeled_data['test'] if data["label"] == 1])
            neg_count = sum([1 for data in labeled_data['test'] if data["label"] == 0])
            logger.debug("test_pos: {}, test_neg: {}".format(pos_count, neg_count))
            labeled_data["metadata"]["test_pos"] = pos_count
            labeled_data["metadata"]["test_neg"] = neg_count
        self.labeled_data = labeled_data

    def llm_annotate_data(self):
        labeled_data = defaultdict(list) # dictionary with 'train' and 'test' fields. Each field is a list of tuples (image_features, label)

        labeled_data_dir = os.path.join(self.config["model_dir"], "labeled_data", self.dataset, self.llm_method)
        labeled_data_path = os.path.join(labeled_data_dir, "udf-{}_run-{}_ntrain-{}_labeled_data.pt".format(self.udf_class, self.run_id, self.n_train_distill))
        os.makedirs(labeled_data_dir, exist_ok=True)
        if self.load_labeled_data and os.path.exists(labeled_data_path):
            logger.info("Loading labeled data from {}".format(labeled_data_path))
            labeled_data = torch.load(labeled_data_path)
            for data in labeled_data['train']:
                logger.debug("row: {}".format(data['row'].drop('img').to_dict()))
                logger.debug("base64_image: {}".format(data["base64_image"]))
                logger.debug("gt_label: {}, llm_label: {}".format(data["label"], data["llm_label"]))
            logger.debug("llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(labeled_data["metadata"]["llm_TP"], labeled_data["metadata"]["llm_FP"], labeled_data["metadata"]["llm_TN"], labeled_data["metadata"]["llm_FN"], labeled_data["metadata"]["llm_f1"]))
            if "test" in labeled_data:
                logger.debug("test_pos: {}, test_neg: {}".format(labeled_data["metadata"]["test_pos"], labeled_data["metadata"]["test_neg"]))
        else:
            self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN = 0, 0, 0, 0
            # Training and validation data
            self.label_count = 0
            for _, row in self.df_train.iterrows():
                logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++")
                try:
                    gt_label = self._get_gt_label(row)
                    # Read and crop frame
                    logger.debug("row: {}".format(row.drop('img').to_dict()))
                    frame, image_size = self.frame_processing_for_model(row)
                    if frame is None:
                        continue
                    # llm_label, base64_image, image_prompt = self._llm_annotate_frame(frame, image_size, row, gt_label)
                    llm_label, base64_image, image_prompt = self._llava_annotate_frame(frame, image_size, row, gt_label)
                    labeled_data['train'].append({"label": gt_label, "llm_label": llm_label, "base64_image": base64_image, "image_prompt": image_prompt, "row": row})
                    if self.label_count >= self.n_train_distill:
                        break
                except Exception as e:
                    logger.exception("Error: {}".format(e))
                    continue
            if self.gt_udf_name is not None:
                llm_f1 = 2*self.llm_TP/(2*self.llm_TP+self.llm_FP+self.llm_FN) if 2*self.llm_TP+self.llm_FP+self.llm_FN > 0 else 0.0
                logger.debug("llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN, llm_f1))
            else:
                llm_f1 = -1
            labeled_data["metadata"] = {"llm_TP": self.llm_TP, "llm_FP": self.llm_FP, "llm_TN": self.llm_TN, "llm_FN": self.llm_FN, "llm_f1": llm_f1}

            # Test data
            if self.gt_udf_name is not None:
                for _, row in self.df_test.iterrows():
                    try:
                        gt_label = self._get_gt_label(row)
                        labeled_data['test'].append({"label": gt_label, "row": row})
                    except Exception as e:
                        logger.exception("Error: {}".format(e))
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

    def mlp_prepare_data(self):
        # TODO: retrieve features directly from parquet files
        splits = ['train', 'test'] if self.gt_udf_name is not None else ['train']
        for split in splits:
            logger.info("Processing {} data".format(split))
            idx_to_remove = []
            for i in range(len(self.labeled_data[split])):
                try:
                    data = self.labeled_data[split][i]
                    row = data['row']
                    logger.debug("row: {}".format(row.drop('img').to_dict()))
                    frame, image_size = self.frame_processing_for_model(row)
                    if frame is None: # failed to read the frame
                        idx_to_remove.append(i)
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_features = self.extract_features(frame, row, image_size)
                    self.labeled_data[split][i]["image_features"] = image_features
                except Exception as e:
                    logger.exception("Error: {}".format(e))
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
        try:
            self.class_weights = torch.tensor([1.0 / count for count in class_counts]).to(self.device)
        except ZeroDivisionError as e:
            logger.exception("Error: {}\nclass_counts: {}".format(e, class_counts))
            self.class_weights = torch.tensor([1.0, 1.0]).to(self.device)

        logger.debug("class_counts: {}, class_weights: {}".format(class_counts, self.class_weights))

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
        if self.gt_udf_name is not None:
            test_dataset = CustomImageDataset(self.labeled_data['test'], train=False)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    def train(self, active_learning_round=-1):
        # logger.debug("mlp_config: {}".format(mlp_config))
        mlp_dim_in = self.dim_in if self.n_obj == 1 else self.dim_in * 3
        logger.debug("mlp_dim_in: {}".format(mlp_dim_in)) # should be 512 for clip-vit-base-patch32
        self.checkpoint_root = os.path.join(self.config["model_dir"], "model_udf", self.dataset, f"{self.llm_method}_{self.mlp_method}", self.udf_class)
        if active_learning_round >= 0:
            self.checkpoint_root = os.path.join(self.checkpoint_root, f"active_learning_round_{active_learning_round}")
        self.checkpoint_filename = "udf-{}_run-{}_ntrain-{}".format(self.udf_class, self.run_id, self.n_train_distill)
        os.makedirs(self.checkpoint_root, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=self.checkpoint_filename,
            monitor="val_loss",
            mode="min",
        )
        callbacks=[checkpoint_callback]
        earlystopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
        callbacks.append(earlystopping_callback)
        # TODO: fine-tuning the learning rate
        # learningrate_callback = pl.callbacks.LearningRateFinder()
        # callbacks.append(learningrate_callback)

        self.mlp_model = mlp.MLP(mlp_dim_in, 2, logger, self.class_weights) # binary classification

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

        # retrieve the best checkpoint after training
        best_ckpt = checkpoint_callback.best_model_path
        logger.debug("Best model checkpoint: {}".format(best_ckpt))
        # best_mlp_model = mlp.MLP.load_from_checkpoint(best_ckpt)
        return best_ckpt

    def test(self):
        # Inference:
        # model = mlp.MLP.load_from_checkpoint(self.dim_in, 2)
        logger.debug("test with last model: ")
        self.trainer.test(self.mlp_model, dataloaders=self.test_loader)
        logger.debug("test with best model: ")
        self.trainer.test(ckpt_path="best", dataloaders=self.test_loader)

    def _get_gt_label(self, row):
        if self.gt_udf_name is None:
            return None
        if self.n_obj == 1:
            return int(self.gt_udf_name in row["o1_gt_anames"])
        else:
            return int(self.gt_udf_name in row["o1_o2_gt_rnames"])

    def frame_processing_for_model(self, row):
        frame = row['img']
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_size = frame.shape[:2]
        if self.n_obj == 1:
            x1, y1, x2, y2 = self.expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
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

    def _llm_annotate_frame(self, frame, image_size, row, gt_label):
        # TODO: don't resize the frame here, resize it in the model
        # Convert the frame to a base 64 encoded image
        # TODO: Try different llm annotation prompt: draw bounding boxes on subject and object.
        # frame = row['img']
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # if self.n_obj == 2:
        #     cv2.rectangle(frame, (int(row['o1_x1']), int(row['o1_y1'])), (int(row['o1_x2']), int(row['o1_y2'])), color=(0, 0, 255), thickness=1)
        #     cv2.rectangle(frame, (int(row['o2_x1']), int(row['o2_y1'])), (int(row['o2_x2']), int(row['o2_y2'])), color=(255, 0, 0), thickness=1)
        # if self.n_obj == 2:
        #     o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
        #     cv2.rectangle(frame, (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
        #     cv2.rectangle(frame, (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(255, 0, 0), thickness=1)
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        logger.debug("base64_image: {}".format(base64_image))
        image_prompt = self._create_image_prompt(row, image_size)
        logger.debug("Image prompt: {}".format(image_prompt))
        response = completion_with_backoff(
            model="gpt-4-turbo-2024-04-09",
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
        logger.debug("gt_label: {}".format(gt_label))
        if "yes" in result.lower():
            llm_label = 1
            if self.gt_udf_name is not None:
                if gt_label == 1:
                    self.llm_TP += 1
                else:
                    self.llm_FP += 1
        elif "no" in result.lower():
            llm_label = 0
            if self.gt_udf_name is not None:
                if gt_label == 0:
                    self.llm_TN += 1
                else:
                    self.llm_FN += 1
        else:
            raise ValueError("Invalid response", result)
        self.label_count += 1
        return llm_label, base64_image, image_prompt

    def _llava_annotate_frame(self, frame, image_size, row, gt_label):
        # Convert the frame to PIL image
        o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
        cv2.rectangle(frame, (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
        cv2.rectangle(frame, (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(255, 0, 0), thickness=1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        llava_image_prompt, last_word = self._create_llava_image_prompt(row, image_size)
        logger.debug("Llava image prompt: {}".format(llava_image_prompt))
        inputs = self.llava_processor(llava_image_prompt, pil_image, return_tensors="pt").to(self.device)
        output = self.llava_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.2,
            top_p=0.7
        )
        result = self.llava_processor.decode(output[0], skip_special_tokens=True)
        result = result.split(last_word)[-1].strip().lower()
        logger.debug("Llava result: {}".format(result))
        logger.debug("gt_label: {}".format(gt_label))
        if "yes" in result.lower():
            llm_label = 1
            if self.gt_udf_name is not None:
                if gt_label == 1:
                    self.llm_TP += 1
                else:
                    self.llm_FP += 1
        elif "no" in result.lower():
            llm_label = 0
            if self.gt_udf_name is not None:
                if gt_label == 0:
                    self.llm_TN += 1
                else:
                    self.llm_FN += 1
        else:
            raise ValueError("Invalid response", result)
        self.label_count += 1
        return llm_label, None, llava_image_prompt

    def _create_image_prompt(self, row, image_size):
        image_prompt = "{}? Answer with 'yes' or 'no'.".format(self.replace_objects(self.udf_description.rstrip(string.punctuation), row, image_size))
        return image_prompt

    def _create_llava_image_prompt(self, row, image_size):
        image_prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{}? Answer with 'yes' or 'no'.<|im_end|><|im_start|>assistant\n".format(self.replace_objects(self.udf_description.rstrip(string.punctuation), row, image_size))
        return image_prompt, "assistant"

    def replace_objects(self, input_string, row, image_size):
        # Find all occurrences of "o" followed by integers
        objects = re.findall(r'o\d+', input_string)
        # Sort the objects based on the integer part of the identifier
        sorted_objects = sorted(set(objects), key=lambda x: int(x[1:]))

        h, w = image_size
        if len(sorted_objects) == 1:
            new_string = input_string.replace(sorted_objects[0], f"{row['o1_oname']} {sorted_objects[0]}")
        elif len(sorted_objects) == 2 and self.n_obj == 2:
            o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
            # o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2']
            # new_string = input_string.replace(sorted_objects[0], f"{row['o1_oname']} {sorted_objects[0]} at {int(o1x1), int(o1y1), int(o1x2), int(o1y2)}")
            # new_string = new_string.replace(sorted_objects[1], f"{row['o2_oname']} {sorted_objects[1]} at {int(o2x1), int(o2y1), int(o2x2), int(o2y2)}")
            new_string = input_string.replace(sorted_objects[0], f"{row['o1_oname']} {sorted_objects[0]} at {int(o1x1), int(o1y1), int(o1x2), int(o1y2)} in the red box")
            new_string = new_string.replace(sorted_objects[1], f"{row['o2_oname']} {sorted_objects[1]} at {int(o2x1), int(o2y1), int(o2x2), int(o2y2)} in the blue box")
        else:
            new_string = input_string

        return new_string

    def extract_features(self, frame, row, image_size):
        """
        three CLIP features: original image, subject mask, target mask
        """
        if self.n_obj == 1:
            inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            outputs = outputs.squeeze(0)
        else:
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


    def construct_train_and_test_data(self, n_obj, n_train=None, n_test=None, df_with_img_column=False, filtered_objects=None):
        if self.dataset == "charades":
            if df_with_img_column:
                return self._construct_train_and_test_data_with_images_person_object_relationships(n_obj, n_train, n_test, filtered_objects=filtered_objects)
            else:
                return self._construct_train_and_test_data_without_images_person_object_relationships(n_obj, n_train, n_test, filtered_objects=filtered_objects)
        else:
            if df_with_img_column:
                return self._construct_train_and_test_data_with_images(n_obj, n_train, n_test)
            else:
                return self._construct_train_and_test_data_without_images(n_obj, n_train, n_test)

    def _construct_train_and_test_data_without_images(self, n_obj, n_train=None, n_test=None):
        # Construct training data and test data
        if n_obj == 1:
            df_filtered = self.one_object_df
        elif n_obj == 2:
            df_filtered = self.two_objects_df
        else:
            raise ValueError("Number of objects not supported: {}".format(n_obj))

        if n_train or n_test:
            df_filtered = df_filtered.sample(n_train + n_test if n_test else n_train, random_state=self.run_id)

        if n_test:
            df_train = df_filtered[:n_train]
            df_train = df_train.reset_index(drop=True)
            df_test = df_filtered[n_train:]
            df_test = df_test.reset_index(drop=True)
            return df_train, df_test
        else:
            df_filtered = df_filtered.reset_index(drop=True)
            return df_filtered

    def _construct_train_and_test_data_with_images(self, n_obj, n_train=None, n_test=None):
        # Construct training data and test data
        # dataframe consists of columns: img, o1 [, o2]
        if n_test:
            df_train, df_test = self._construct_train_and_test_data_without_images(n_obj, n_train, n_test)
            # construct the img column
            for df in [df_train, df_test]:
                df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
            return df_train, df_test
        else:
            df = self._construct_train_and_test_data_without_images(n_obj, n_train)
            df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
            return df

    def _construct_train_and_test_data_without_images_person_object_relationships(self, n_obj, n_train=None, n_test=None, filtered_objects=None):
        n_samples = n_train + n_test if n_test else n_train
        # Construct training data and test data
        if n_obj == 1: # Charades shouldn't need this, but just in case
            df_filtered = self.one_object_df
        elif n_obj == 2: # Only consider person-object relationships
            # vid, fid, o1_oid, o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oid, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, o1_o2_gt_rnames, height, width
            if filtered_objects:
                # obj_parameters = ','.join('?' for _ in filtered_objects)
                df_filtered = self.two_objects_df[((self.two_objects_df["o1_oid"] == 0) | (self.two_objects_df["o2_oid"] == 0)) & (self.two_objects_df["o1_oname"].isin(filtered_objects)) & (self.two_objects_df["o2_oname"].isin(filtered_objects))]
                # self.conn.execute(f"""
                #     SELECT *
                #     FROM self.two_objects_df
                #     WHERE (o1_oid = 0 OR o2_oid = 0) AND o1_oname = ANY([{obj_parameters}]) AND o2_oname = ANY([{obj_parameters}])
                # """, filtered_objects + filtered_objects).df()
            else:
                df_filtered = self.two_objects_df[(self.two_objects_df["o1_oid"] == 0) | (self.two_objects_df["o2_oid"] == 0)]

            if len(df_filtered) < n_samples:
                logger.warning(f"Number of samples ({len(df_filtered)}) is less than n_samples ({n_samples}).")
                df_filtered = self.two_objects_df
        else:
            raise ValueError("Number of objects not supported: {}".format(n_obj))

        if n_train or n_test:
            df_filtered = df_filtered.sample(n_samples, random_state=self.run_id)

        if n_test:
            df_train = df_filtered[:n_train]
            df_train = df_train.reset_index(drop=True)
            df_test = df_filtered[n_train:]
            df_test = df_test.reset_index(drop=True)
            return df_train, df_test
        else:
            df_filtered = df_filtered.reset_index(drop=True)
            return df_filtered

    def _construct_train_and_test_data_with_images_person_object_relationships(self, n_obj, n_train=None, n_test=None, filtered_objects=None):
        # Construct training data and test data
        # dataframe consists of columns: img, o1 [, o2]
        if n_test:
            df_train, df_test = self._construct_train_and_test_data_without_images_person_object_relationships(n_obj, n_train, n_test, filtered_objects=filtered_objects)
            # construct the img column
            for df in [df_train, df_test]:
                df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
            return df_train, df_test
        else:
            df = self._construct_train_and_test_data_without_images_person_object_relationships(n_obj, n_train, filtered_objects=filtered_objects)
            df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
            return df

    def frame_processing_for_program(self, vid, fid):
        if self.dataset == "clevrer":
            frame = np.array(Image.open(os.path.join(
                self.config[self.dataset]["video_frames_dir"],
                f"sim_{str(vid).zfill(5)}",
                f"frame_{str(fid).zfill(5)}.png"
            ))) # Shape: (H, W, C)
        elif self.dataset == "charades":
            frame = np.array(Image.open(os.path.join(
                self.config[self.dataset]["video_frames_dir"],
                f"{self.vid_to_vname[vid]}.mp4",
                f"{str(fid).zfill(6)}.png"
            ))) # Shape: (H, W, C)
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")
        return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--udf_idx", type=int, help="udf_idx")

    args = parser.parse_args()
    udf_idx = args.udf_idx
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )
    n_train_distill = 1000
    test_inputs = [
        ["not_looking_at(o0, o1)", "Whether o0 is not looking at o1. Return True if o0 is not looking at o1, otherwise return False.", "not_looking_at"],
        ["looking_at(o0, o1)", "Whether object o0 is looking at object o1", "looking_at"],
        ["holding(o0, o1)", "Whether o0 is holding o1", "holding"],
        ["not_contacting(o0, o1)", "Whether o0 is not contacting o1. Return True if o0 is not contacting o1, otherwise return False.", "not_contacting"],
        ["touching(o0, o1)", "Whether o0 is touching o1.", "touching"],
        ["sitting_on(o0, o1)", "Whether o0 is sitting on o1.", "sitting_on"],
        ["leaning_on(o0, o1)", "Whether o0 is leaning on o1.", "leaning_on"],
        ["drinking_from(o0, o1)", "Whether o0 is drinking from o1.", "drinking_from"],
        ["standing_on(o0, o1)", "Whether o0 is standing on o1.", "standing_on"],
        ["wearing(o0, o1)", "Whether o0 is wearing o1.", "wearing"],
    ]
    test_input = test_inputs[udf_idx]
    """
    Set up logging
    """
    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(f"/gscratch/balazinska/enhaoz/VOCAL-UDF/tests/gt-ntrain_distill={n_train_distill}-udf_idx={udf_idx}-llava.log", mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler with a higher log level
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    # logger.addHandler(console_handler)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    sys.excepthook = exception_hook

    dataset = "charades"
    registered_functions = [{
        "signature": "object(o0, name)",
        "description": "Whether o0 is an object with the given name.",
        "function_implementation": ""
    }]
    object_domain, relationship_domain, attribute_domain = get_active_domain(config, dataset, registered_functions)
    query_id = 0
    run_id = 0
    num_workers = 8
    save_labeled_data = False
    load_labeled_data = False
    up = UDFProposer(
        config,
        prompt_config,
        object_domain,
        relationship_domain,
        attribute_domain,
        dataset,
        query_id,
        run_id,
        num_workers,
        save_labeled_data,
        load_labeled_data,
        n_train_distill,
    )
    up._distill_model(test_input[0], test_input[1], test_input[2])