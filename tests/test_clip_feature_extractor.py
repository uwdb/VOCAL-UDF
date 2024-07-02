import argparse
import os
from vocaludf.utils import (
    replace_slot,
    parse_signature,
    PredImageDataset,
)
from vocaludf.udf_proposer import UDFProposer, UDFCandidate, CustomImageDataset
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
from sklearn.metrics import f1_score, confusion_matrix
import copy
import cv2
from tqdm import tqdm
import sys
from transformers import CLIPProcessor, CLIPModel, LlavaNextForConditionalGeneration, LlavaNextProcessor, CLIPTokenizer
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

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

class TestDistillModel(UDFProposer):
    def _distill_model(self, udf_signature, udf_description, gt_udf_name, candidate_classes, n_train):
        """
        gt_udf_name: ground truth UDF name. If provided, compute llm's TP, FP, TN, FN, f1 score and test the trained model.
        """
        self.attribute_df = self.conn.execute(f"SELECT * FROM {self.dataset}_attributes").df()
        self.relationship_df = self.conn.execute(f"SELECT * FROM {self.dataset}_relationships").df()

        clip_model_name = os.path.join(self.config['model_dir'], 'clip-vit-base-patch32')
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        # Initialization for model distillation
        udf_name, udf_vars = parse_signature(udf_signature)
        self.udf_class = udf_name.lower()
        self.n_obj = len(udf_vars)
        assert self.n_obj in [1, 2], "n_obj must be 1 or 2"
        self.udf_description = udf_description

        self.gt_udf_name = gt_udf_name
        self.candidate_classes = candidate_classes
        self.texts = [f"a photo of a {c} object" for c in candidate_classes]
        self.df_train = self.construct_data(self.n_obj, n_train)
        self.clip_predict()

    def clip_predict(self):
        # get text embedding
        text_embeddings = []
        for text in self.texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_embedding = self.clip_model.get_text_features(**inputs)
            # convert the embeddings to numpy array
            text_embedding = text_embedding.cpu().detach().numpy()
            text_embeddings.append(text_embedding)
        text_embeddings = np.concatenate(text_embeddings, axis=0)
        text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True) # shape: (n_texts, 512)

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
                if frame is None: # failed to read the frame
                    continue
                _, buffer = cv2.imencode('.jpg', frame)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                logger.debug("base64_image: {}".format(base64_image))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_features = self.extract_features(frame, row, image_size)
                image_features = image_features.cpu().detach().numpy()
                # Compute the similarity between the image and the text
                image_features = image_features / np.linalg.norm(image_features)
                similarities = np.dot(text_embeddings, image_features)
                logger.debug("similarities: {}".format(similarities))
                sorted_indices = np.argsort(similarities)[::-1]
                logger.debug("sorted_classes: {}".format([self.candidate_classes[i] for i in sorted_indices]))

                # find the text with the highest similarity
                max_idx = np.argmax(similarities)
                logger.debug("max_idx: {}, pred_class: {}".format(max_idx, self.candidate_classes[max_idx]))
                pred = int(self.candidate_classes[max_idx] == self.gt_udf_name)
                logger.debug("pred: {}".format(pred))
                logger.debug("gt_label: {}".format(gt_label))
                if gt_label == 1 and pred == 1:
                    self.llm_TP += 1
                elif gt_label == 0 and pred == 1:
                    self.llm_FP += 1
                elif gt_label == 0 and pred == 0:
                    self.llm_TN += 1
                elif gt_label == 1 and pred == 0:
                    self.llm_FN += 1
            except Exception as e:
                logger.exception("Error: {}".format(e))
                continue
        llm_f1 = 2*self.llm_TP/(2*self.llm_TP+self.llm_FP+self.llm_FN) if 2*self.llm_TP+self.llm_FP+self.llm_FN > 0 else 0.0
        logger.debug("TP: {}, FP: {}, TN: {}, FN: {}, f1: {}".format(self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN, llm_f1))

    def construct_data(self, n_obj, n_train=None):
        if self.dataset == "charades":
            return self._construct_train_and_test_data_with_images_person_object_relationships(n_obj, n_train)
        else:
            return self._construct_data_with_images(n_obj, n_train)

    def _construct_data_with_images(self, n_obj, n_train=None):
        # Construct training data and test data
        # dataframe consists of columns: img, o1 [, o2]
        df = self._construct_data_without_images(n_obj, n_train)
        df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
        return df

    def _construct_data_without_images(self, n_obj, n_train=None):
        # Construct training data and test data
        if n_obj == 1:
            pos_sql = """
                SELECT
                    o1.vid AS vid, o1.fid AS fid,
                    o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                    ARRAY[a1.aname] AS o1_gt_anames
                FROM {}_metadata m, {}_objects o1, {}_attributes a1
                WHERE m.vid = o1.vid AND m.fid = o1.fid AND o1.vid = a1.vid AND o1.fid = a1.fid AND o1.oid = a1.oid AND a1.aname = '{}'
                ORDER BY random()
                LIMIT {}
            """.format(
                self.dataset, self.dataset, self.dataset, self.udf_class, n_train // 2
            )
            neg_sql = """
                SELECT
                    o1.vid AS vid, o1.fid AS fid,
                    o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                    ARRAY[]::varchar[] as o1_gt_anames
                FROM {}_metadata m, {}_objects o1
                WHERE m.vid = o1.vid AND m.fid = o1.fid
                    AND NOT EXISTS (
                        SELECT 1
                        FROM {}_attributes a1
                        WHERE a1.vid = o1.vid AND a1.fid = o1.fid AND a1.oid = o1.oid AND a1.aname = '{}'
                    )
                ORDER BY random()
                LIMIT {}
            """.format(
                self.dataset, self.dataset, self.dataset, self.udf_class, n_train // 2
            )
        elif n_obj == 2:
            pos_sql = """
                SELECT
                    o1.vid AS vid, o1.fid AS fid,
                    o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                    o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2,
                    ARRAY[r.rname] as o1_o2_gt_rnames
                FROM {}_metadata m, {}_objects o1, {}_objects o2, {}_relationships r
                WHERE m.vid = o1.vid AND m.fid = o1.fid AND o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
                    AND o1.vid = r.vid AND o1.fid = r.fid AND o1.oid = r.oid1 AND o2.oid = r.oid2 AND r.rname = '{}'
                ORDER BY random()
                LIMIT {}
            """.format(
                self.dataset, self.dataset, self.dataset, self.dataset, self.udf_class, n_train // 2
            )
            neg_sql = """
                SELECT
                    o1.vid AS vid, o1.fid AS fid,
                    o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                    o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2,
                    ARRAY[]::varchar[] as o1_o2_gt_rnames
                FROM {}_metadata m, {}_objects o1, {}_objects o2
                WHERE m.vid = o1.vid AND m.fid = o1.fid AND o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
                    AND NOT EXISTS (
                        SELECT 1
                        FROM {}_relationships r
                        WHERE r.vid = o1.vid AND r.fid = o1.fid AND r.oid1 = o1.oid
                            AND r.oid2 = o2.oid AND r.rname = '{}'
                    )
                ORDER BY random()
                LIMIT {}
            """.format(
                self.dataset, self.dataset, self.dataset, self.dataset, self.udf_class, n_train // 2
            )
        else:
            raise ValueError("Number of objects not supported: {}".format(n_obj))

        # logger.debug("pos_sql: {}".format(pos_sql))
        # logger.debug("neg_sql: {}".format(neg_sql))
        df_pos = self.conn.execute(pos_sql).df()
        df_neg = self.conn.execute(neg_sql).df()

        # Interleave the rows of the two DataFrames
        df_filtered = pd.concat([df_pos, df_neg], ignore_index=True)
        max_length = max(len(df_pos), len(df_neg))
        new_index = np.array([[i, i + max_length] for i in range(max_length)]).flatten()
        new_index = new_index[new_index < len(df_filtered)]
        df_filtered = df_filtered.iloc[new_index]

        df_filtered = df_filtered.reset_index(drop=True)
        return df_filtered


if __name__ == "__main__":
    # python test_clip_feature_extractor.py --dataset "vaw" --udf_idx 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--udf_idx", type=int, help="udf_idx")

    args = parser.parse_args()
    dataset = args.dataset
    udf_idx = args.udf_idx

    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )

    if dataset == "charades":
        test_inputs = [
            ["holding(o0, o1)", "Whether o0 is holding o1.", "holding"],
            ["sitting_on(o0, o1)", "Whether o0 is sitting on o1.", "sitting_on"],
            ["standing_on(o0, o1)", "Whether o0 is standing on o1.", "standing_on"],
            ["covered_by(o0, o1)", "Whether o0 is covered by o1.", "covered_by"],
            ["carrying(o0, o1)", "Whether o0 is carrying o1.", "carrying"],
            ["eating(o0, o1)", "Whether o0 is eating o1.", "eating"],
            ["wiping(o0, o1)", "Whether object o0 is wiping o1", "wiping"],
            ["have_it_on_the_back(o0, o1)", "Whether o0 has o1 on the back.", "have_it_on_the_back"],
            ["leaning_on(o0, o1)", "Whether o0 is leaning on o1.", "leaning_on"],
            ["wearing(o0, o1)", "Whether o0 is wearing o1.", "wearing"],
        ]
    elif dataset == "vaw":
        test_inputs = [
            ["white(o0)", "Whether the color of o0 is white.", "white", ["white", "green", "red", "yellow", "silver", "black", "blue", "brown", "gray", "orange"]],
            ["green(o0)", "Whether the color of o0 is green.", "green", ["white", "green", "red", "yellow", "silver", "black", "blue", "brown", "gray", "orange"]],
            ["large(o0)", "Whether the size of o0 is large.", "large", ["small", "large"]],
            ["red(o0)", "Whether the color of o0 is red.", "red", ["white", "green", "red", "yellow", "silver", "black", "blue", "brown", "gray", "orange"]],
            ["wooden(o0)", "Whether the material of o0 is wooden.", "wooden", ["wooden", "metal", "rubber"]],
            ["yellow(o0)", "Whether the color of o0 is yellow.", "yellow", ["white", "green", "red", "yellow", "silver", "black", "blue", "brown", "gray", "orange"]],
            ["tall(o0)", "Whether the height of o0 is tall.", "tall", ["tall", "short"]],
            ["silver(o0)", "Whether the color of o0 is silver.", "silver", ["white", "green", "red", "yellow", "silver", "black", "blue", "brown", "gray", "orange"]],
            ["standing(o0)", "Whether the pose of o0 is standing.", "standing", ["standing", "sitting", "lying"]],
            ["round(o0)", "Whether the shape of o0 is round.", "round", ["round", "square", "triangular"]],
        ]

    test_input = test_inputs[udf_idx]
    """
    Set up logging
    """
    # Create a file handler that logs even debug messages
    log_dir = os.path.join(
        config["log_dir"],
        "clip_feature_extractor",
        dataset
    )
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"udf_idx={udf_idx}.log"), mode="w")
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

    registered_functions = [{
        "signature": "object(o0, name)",
        "description": "Whether o0 is an object with the given name.",
        "function_implementation": ""
    }]
    object_domain, relationship_domain, attribute_domain = get_active_domain(config, dataset, registered_functions)
    num_workers = 8

    up = TestDistillModel(
            config,
            prompt_config,
            registered_functions,
            object_domain,
            relationship_domain,
            attribute_domain,
            dataset,
            0,
            500,
            0,
            0,
            False,
            False,
            0,
            0,
            num_workers,
            False,
            False,
            -1,
            selection_strategy='model',
            selection_labels='none',
            allow_kwargs_in_udf=False,
            llm_method='user',
        )
    up._distill_model(*test_input, n_train=1000)