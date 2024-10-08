from typing_extensions import Annotated
import autogen
from autogen import gather_usage_summary
import json
from typing import Tuple, List
import os
import math
from enum import Enum
from vocaludf.utils import (
    replace_slot,
    parse_signature,
    transform_function,
    PredImageDataset,
    MODEL_COST,
    RESOLVE_MODEL_NAME,
    expand_box,
)
from vocaludf.pretrained_model_api import image_captioning, visual_question_answering, depth_estimation
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import partial
from PIL import Image
import time
import signal
import resource
import duckdb
import logging
from openai import OpenAI, AsyncOpenAI
import re
from collections import defaultdict
import importlib
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import copy
import cv2
from tqdm import tqdm
import sys
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, LlavaNextForConditionalGeneration, LlavaNextProcessor, AlignModel, AlignProcessor
import torch
from torch.utils.data import Dataset
import base64
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    after_log,
)  # for exponential backoff
import string
import lightning.pytorch as pl
from vocaludf import mlp
import torchvision.ops as ops
import torchvision.transforms as T
import asyncio

tqdm.pandas()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SamplingStrategy = Enum('SamplingStrategy', ['positive', 'negative', 'uncertainty'])

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

class UDFCandidate:
    def __init__(self, id, payload):
        self.kwargs = payload.get("kwargs", {})
        self.id = str(id) + "_" + "_".join([f"{k}-{v}" for k, v in self.kwargs.items()]) if self.kwargs else str(id) # 'model' for model-based UDFs
        self.udf_name = payload["udf_name"]
        self.udf_signature = payload["udf_signature"]
        self.udf_description = payload["udf_description"]
        self.semantic_interpretation = payload["semantic_interpretation"] # 'model' for model-based UDFs
        self.function_implementation = payload["function_implementation"] # python function for program-based UDFs, and path_to_best_ckpt for model-based UDFs
        self.score = 1  # F1 score
        self.test_score = -1
        self.loss_t = 0  # loss_t = n_misclassified

    def __str__(self):
        return f"UDFCandidate(id: {self.id}, function_implementation: {self.function_implementation}, test_score: {self.test_score}, score: {self.score}, loss_t: {self.loss_t})"

class SharedResources:
    def __init__(
        self,
        config,
        prompt_config,
        registered_functions,
        object_domain,
        relationship_domain,
        attribute_domain,
        dataset,
        labeling_budget,
        n_selection_samples,
        num_interpretations,
        num_parameter_search,
        program_with_pixels,
        program_with_pretrained_models,
        query_class_name,
        query_id,
        run_id,
        num_workers,
        save_labeled_data,
        load_labeled_data,
        n_train_distill,
        selection_strategy,
        selection_labels,
        allow_kwargs_in_udf,
        llm_method,
        is_async,
        openai_model_name
    ):
        self.config = config
        self.prompt_config = prompt_config
        self.registered_functions = registered_functions
        self.object_domain = object_domain
        self.relationship_domain = relationship_domain
        self.attribute_domain = attribute_domain
        self.dataset = dataset
        self.labeling_budget = labeling_budget
        self.n_selection_samples = n_selection_samples
        self.num_interpretations = num_interpretations
        self.num_parameter_search = num_parameter_search
        self.program_with_pixels = program_with_pixels
        self.program_with_pretrained_models = program_with_pretrained_models
        self.query_class_name = query_class_name
        self.query_id = query_id
        self.run_id = run_id
        self.num_workers = num_workers
        self.selection_strategy = selection_strategy
        self.selection_labels = selection_labels
        self.allow_kwargs_in_udf = allow_kwargs_in_udf
        self.llm_method = llm_method
        self.is_async = is_async
        self.openai_model_name = RESOLVE_MODEL_NAME[openai_model_name]
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )

        # Create a train and test split
        # NOTE: probably put these values in the config file
        # cityflow has higher resolution images, so we use fewer samples
        self.n_train = 5000 if self.dataset == "cityflow" else 10000
        self.n_test = 5000 if self.dataset == "cityflow" else 10000

        # Initialization for model distillation
        self.n_train_distill = n_train_distill
        self.n_test_distill = 1000
        self.save_labeled_data = save_labeled_data
        self.load_labeled_data = load_labeled_data
        self.attribute_features_dir = os.path.join(self.config[self.dataset]["features_dir"], "attribute")
        self.relationship_features_dir = os.path.join(self.config[self.dataset]["features_dir"], "relationship")

        logger.info("Table initialization started")
        self.one_object_df, self.two_objects_df = self.init_table()
        logger.info("Table initialization finished")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the CLIP model
        clip_model_name = os.path.join(self.config['model_dir'], 'clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = None
        if self.dataset == "charades":
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        # clip_model_name = "openai/clip-vit-base-patch32"
        # self.clip_model.save_pretrained(os.path.join(self.config['model_dir'], 'clip-vit-base-patch32'))
        # self.clip_processor.save_pretrained(os.path.join(self.config['model_dir'], 'clip-vit-base-patch32'))
        # clip_model_name = os.path.join(self.config['model_dir'], 'align-base')
        # self.clip_model = AlignModel.from_pretrained(clip_model_name).to(self.device)
        # self.clip_processor = AlignProcessor.from_pretrained(clip_model_name)
        # self.clip_model.save_pretrained(os.path.join(self.config['model_dir'], 'align-base'))
        # self.clip_processor.save_pretrained(os.path.join(self.config['model_dir'], 'align-base'))

        self.dim_in = self.clip_model.config.projection_dim

        self.llava_model = None
        self.llava_processor = None
        if self.llm_method == "llava":
            # Make room for llava model
            llava_model_name = os.path.join(self.config['model_dir'], 'llava-v1.6-34b-hf')
            self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                llava_model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            logger.debug("llava_model.hf_device_map: {}".format(self.llava_model.hf_device_map))
            self.llava_processor = LlavaNextProcessor.from_pretrained(llava_model_name)
            self.llava_processor.tokenizer.padding_side = "left"

        self.vid_to_vname = None
        if self.dataset == "charades":
            df_metadata = self.conn.execute(f"""
                SELECT DISTINCT vname, vid
                FROM charades_metadata
            """).df()
            self.vid_to_vname = {int(vid): vname for vid, vname in zip(df_metadata['vid'], df_metadata['vname'])}
        elif self.dataset == "cityflow":
            df_metadata = self.conn.execute(f"""
                SELECT vname, vid, fid
                FROM cityflow_metadata
            """).df()
            self.vid_to_vname = {(vid, fid): vname for vid, fid, vname in zip(df_metadata['vid'], df_metadata['fid'], df_metadata['vname'])}

    def init_table(self):
        metadata_join_clause = '' if self.dataset in ['clevr', 'clevrer'] else f'LEFT OUTER JOIN {self.dataset}_metadata m ON o1.vid = m.vid AND o1.fid = m.fid'
        height_width_clause = '320 AS height, 480 AS width' if self.dataset in ['clevr', 'clevrer'] else 'm.height AS height, m.width AS width'
        group_by_clause = 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2' if self.dataset in ['clevr', 'clevrer'] else 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2, m.height, m.width'
        obj_parameters = ','.join('?' for _ in self.object_domain)
        attr_parameters = ','.join('?' for _ in self.attribute_domain)
        if self.dataset == 'vaw':
            where_clause = ""
        elif self.dataset == 'clevrer':
            where_clause = f"WHERE o1.oname = ANY([{obj_parameters}]) AND o1.vid < 5000"
        elif self.dataset == 'charades':
            where_clause = f"WHERE o1.oname = ANY([{obj_parameters}]) AND o1.vid < 4800"
        elif self.dataset == 'cityflow':
            where_clause = f"WHERE o1.oname = ANY([{obj_parameters}]) AND o1.vid < 824"
        else:
            where_clause = f"WHERE o1.oname = ANY([{obj_parameters}])"
        select_neg_attr_clause = f"COALESCE(ARRAY_AGG(DISTINCT a2.aname) FILTER (WHERE a2.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames_negative," if self.dataset == 'vaw' else ""
        join_neg_attr_clause = f"LEFT OUTER JOIN {self.dataset}_attributes_negative a2 ON o1.vid = a2.vid AND o1.fid = a2.fid AND o1.oid = a2.oid" if self.dataset == 'vaw' else ""
        sql = f"""
            SELECT
                o1.vid AS vid, o1.fid AS fid, o1.oid AS o1_oid, o1.oname AS o1_oname,
                o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                COALESCE(ARRAY_AGG(DISTINCT a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
                {select_neg_attr_clause}
                COALESCE(ARRAY_AGG(DISTINCT ap.aname) FILTER (WHERE ap.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
                {height_width_clause}
            FROM {self.dataset}_objects o1
            LEFT OUTER JOIN {self.dataset}_attributes a ON o1.vid = a.vid AND o1.fid = a.fid AND o1.oid = a.oid
            LEFT OUTER JOIN {self.dataset}_attribute_predictions ap ON o1.vid = ap.vid AND o1.fid = ap.fid AND o1.oid = ap.oid
            {join_neg_attr_clause}
            {metadata_join_clause}
            {where_clause}
            GROUP BY {group_by_clause}
        """
        # ORDER BY o1.vid, o1.fid, o1.oid, o1.x1, o1.y1, o1.x2, o1.y2
        logger.debug(f"Create one_object table:\n{sql}")
        one_object_df = self.conn.execute(sql, self.attribute_domain if self.dataset == 'vaw' else self.attribute_domain + self.object_domain).df()

        rel_parameters = ','.join('?' for _ in self.relationship_domain)
        where_clause = "" if self.dataset == 'vaw' else f"WHERE o.oname = ANY([{obj_parameters}])"
        if self.dataset == 'clevrer':
            filter_train_vid_clause = "AND o1.vid < 5000"
        elif self.dataset == 'charades':
            filter_train_vid_clause = "AND o1.vid < 4800"
        elif self.dataset == 'cityflow':
            filter_train_vid_clause = "AND o1.vid < 824"
        else:
            filter_train_vid_clause = ""
        sql = f"""
            WITH obj_with_attrs AS (
                SELECT
                    o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                    COALESCE(ARRAY_AGG(DISTINCT a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS attributes
                FROM {self.dataset}_objects o
                LEFT OUTER JOIN {self.dataset}_attribute_predictions a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid AND a.aname = ANY([{attr_parameters}])
                {where_clause}
                GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
            )
            , relationships_expanded AS (
                SELECT
                    vid, fid, oid1, oid2,
                    ARRAY_AGG(DISTINCT rname) AS gt_rnames
                FROM {self.dataset}_relationships
                GROUP BY vid, fid, oid1, oid2
            )
            , relationship_predictions_expanded AS (
                SELECT
                    vid, fid, oid1, oid2,
                    COALESCE(ARRAY_AGG(DISTINCT rname) FILTER (WHERE rname = ANY([{rel_parameters}])), ARRAY[]::varchar[]) AS rnames
                FROM {self.dataset}_relationship_predictions
                GROUP BY vid, fid, oid1, oid2
            )
            SELECT
                o1.vid AS vid, o1.fid AS fid,
                o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2, o1.attributes AS o1_anames,
                o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2, o2.attributes AS o2_anames,
                COALESCE(r1.rnames, ARRAY[]::varchar[]) AS o1_o2_rnames,
                COALESCE(r2.rnames, ARRAY[]::varchar[]) AS o2_o1_rnames,
                COALESCE(r3.gt_rnames, ARRAY[]::varchar[]) AS o1_o2_gt_rnames,
                {height_width_clause}
            FROM obj_with_attrs o1
            JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid
            LEFT OUTER JOIN relationship_predictions_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
            LEFT OUTER JOIN relationship_predictions_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
            LEFT OUTER JOIN relationships_expanded r3 ON o1.vid = r3.vid AND o1.fid = r3.fid AND o1.oid = r3.oid1 AND o2.oid = r3.oid2
            {metadata_join_clause}
            WHERE o1.oid <> o2.oid
            {filter_train_vid_clause}
        """
        # ORDER BY o1.vid, o1.fid, o1.oid, o2.oid, o1.x1, o1.y1, o1.x2, o1.y2, o2.x1, o2.y1, o2.x2, o2.y2
        logger.debug(f"Create two_objects table:\n{sql}")
        two_objects_df = self.conn.execute(sql, self.attribute_domain + self.relationship_domain if self.dataset == 'vaw' else self.attribute_domain + self.object_domain + self.relationship_domain).df()
        return one_object_df, two_objects_df


class UtilsMixin:
    def exec_udf_with_data(self, df, udf_obj, kwargs, n_obj, requires_no_error=True, timeout=None):
        def safe_udf(udf, *args, **kwargs):
            try:
                return bool(udf(*args, **kwargs))
            except Exception as e:
                logger.exception(f"exec_udf_with_data Error: {e}")
                return False  # Default value in case of error

        if requires_no_error:
            func = partial(udf_obj, **kwargs)
        else:
            func = partial(safe_udf, udf_obj, **kwargs)

        if n_obj == 1:
            if self.program_with_pixels:
                args = (df["img"], df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["height"], df["width"])
            else:
                args = (df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["height"], df["width"])
        elif n_obj == 2:
            if self.program_with_pixels:
                args = (df["img"], df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["o2_oname"], df["o2_x1"], df["o2_y1"], df["o2_x2"], df["o2_y2"], df["o2_anames"], df["o1_o2_rnames"], df["o2_o1_rnames"], df["height"], df["width"])
            else:
                args = (df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["o2_oname"], df["o2_x1"], df["o2_y1"], df["o2_x2"], df["o2_y2"], df["o2_anames"], df["o1_o2_rnames"], df["o2_o1_rnames"], df["height"], df["width"])

        result = list(tqdm(self.executor.map(func, *args, timeout=timeout), total=len(df), file=sys.stdout, desc="exec_udf_with_data"))

        return result


    def _compute_new_box_after_crop(self, row, image_size):
        o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
        o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
        x_offset = min(o1_x1, o2_x1)
        y_offset = min(o1_y1, o2_y1)
        h_ratio = 224.0 / (max(o1_y2, o2_y2) - y_offset)
        w_ratio = 224.0 / (max(o1_x2, o2_x2) - x_offset)
        return (row['o1_x1'] - x_offset) * w_ratio, (row['o1_y1'] - y_offset) * h_ratio, (row['o1_x2'] - x_offset) * w_ratio, (row['o1_y2'] - y_offset) * h_ratio, (row['o2_x1'] - x_offset) * w_ratio, (row['o2_y1'] - y_offset) * h_ratio, (row['o2_x2'] - x_offset) * w_ratio, (row['o2_y2'] - y_offset) * h_ratio


    def construct_train_and_test_data(self, n_obj, n_train=None, n_test=None, df_with_img_column=False, filtered_objects=None, filtered_subjects=None, filtered_targets=None, gt_udf_name=None):
        if self.dataset == "charades":
            if df_with_img_column:
                return self._construct_train_and_test_data_with_images_person_object_relationships(n_obj, n_train, n_test, filtered_objects=filtered_objects)
            else:
                return self._construct_train_and_test_data_without_images_person_object_relationships(n_obj, n_train, n_test, filtered_objects=filtered_objects)
        elif self.dataset == "vaw" and gt_udf_name: # Only sample from positive and negative examples
            if df_with_img_column:
                return self._construct_train_and_test_data_with_images_vaw(gt_udf_name, n_obj, n_train, n_test)
            else:
                return self._construct_train_and_test_data_without_images_vaw(gt_udf_name, n_obj, n_train, n_test)
        else:
            if df_with_img_column:
                return self._construct_train_and_test_data_with_images(n_obj, n_train, n_test, filtered_objects=filtered_objects, filtered_subjects=filtered_subjects, filtered_targets=filtered_targets)
            else:
                return self._construct_train_and_test_data_without_images(n_obj, n_train, n_test, filtered_objects=filtered_objects, filtered_subjects=filtered_subjects, filtered_targets=filtered_targets)

    def _construct_train_and_test_data_without_images(self, n_obj, n_train=None, n_test=None, filtered_objects=None, filtered_subjects=None, filtered_targets=None):
        n_samples = n_train + n_test if n_test else n_train
        # Construct training data and test data
        if n_obj == 1:
            if filtered_objects:
                df_filtered = self.one_object_df[self.one_object_df["o1_oname"].isin(filtered_objects)]
            else:
                df_filtered = self.one_object_df

            if len(df_filtered) < n_samples:
                logger.warning(f"Number of samples ({len(df_filtered)}) is less than n_samples ({n_samples}).")
                df_filtered = self.one_object_df
        elif n_obj == 2:
            if filtered_subjects and filtered_targets:
                df_filtered = self.two_objects_df[(self.two_objects_df["o1_oname"].isin(filtered_subjects)) & (self.two_objects_df["o2_oname"].isin(filtered_targets))]
            elif filtered_subjects:
                df_filtered = self.two_objects_df[self.two_objects_df["o1_oname"].isin(filtered_subjects)]
            elif filtered_targets:
                df_filtered = self.two_objects_df[self.two_objects_df["o2_oname"].isin(filtered_targets)]
            else:
                df_filtered = self.two_objects_df

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

    def _construct_train_and_test_data_without_images_vaw(self, gt_udf_name, n_obj, n_train=None, n_test=None):
        n_samples = n_train + n_test if n_test else n_train
        # Construct training data and test data
        if n_obj == 1:
            # vid, fid, o1_oid, o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_gt_anames, o1_gt_anames_negative, o1_anames, height, width
            # select rows where gt_udf_name is in o1_gt_anames or o1_gt_anames_negative
            df_filtered = self.one_object_df[(self.one_object_df["o1_gt_anames"].apply(lambda x: gt_udf_name in x)) | (self.one_object_df["o1_gt_anames_negative"].apply(lambda x: gt_udf_name in x))]
        elif n_obj == 2:
            # vid, fid, o1_oid, o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oid, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, o1_o2_gt_rnames, height, width
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

    def _construct_train_and_test_data_with_images_vaw(self, gt_udf_name, n_obj, n_train=None, n_test=None):
        # Construct training data and test data
        # dataframe consists of columns: img, o1 [, o2]
        if n_test:
            df_train, df_test = self._construct_train_and_test_data_without_images_vaw(gt_udf_name, n_obj, n_train, n_test)
            # construct the img column
            for df in [df_train, df_test]:
                df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
            return df_train, df_test
        else:
            df = self._construct_train_and_test_data_without_images_vaw(gt_udf_name, n_obj, n_train)
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
                df_filtered = self.two_objects_df[(self.two_objects_df["o1_oid"] == 0) & (self.two_objects_df["o1_oname"].isin(filtered_objects)) & (self.two_objects_df["o2_oname"].isin(filtered_objects))]
                # df_filtered = self.two_objects_df[((self.two_objects_df["o1_oid"] == 0) | (self.two_objects_df["o2_oid"] == 0)) & (self.two_objects_df["o1_oname"].isin(filtered_objects)) & (self.two_objects_df["o2_oname"].isin(filtered_objects))]
            else:
                df_filtered = self.two_objects_df[(self.two_objects_df["o1_oid"] == 0)]

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

    def _construct_train_and_test_data_with_images(self, n_obj, n_train=None, n_test=None, filtered_objects=None, filtered_subjects=None, filtered_targets=None):
        # Construct training data and test data
        # dataframe consists of columns: img, o1 [, o2]
        if n_test:
            df_train, df_test = self._construct_train_and_test_data_without_images(n_obj, n_train, n_test, filtered_objects=filtered_objects, filtered_subjects=filtered_subjects, filtered_targets=filtered_targets)
            # construct the img column
            for df in [df_train, df_test]:
                df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
            return df_train, df_test
        else:
            df = self._construct_train_and_test_data_without_images(n_obj, n_train, filtered_objects=filtered_objects, filtered_subjects=filtered_subjects, filtered_targets=filtered_targets)
            df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
            return df

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
        elif self.dataset in ["gqa", "vaw"]:
            frame = np.array(Image.open(os.path.join(
                self.config[self.dataset]["video_frames_dir"],
                f"{vid % 10}",
                f"{vid}.jpg"
            ))) # Shape: (H, W, C)
            if len(frame.shape) == 2:
                frame = np.stack([frame] * 3, axis=-1)
        elif self.dataset == "cityflow":
            vname = self.vid_to_vname[(vid, fid)]
            image_file = os.path.join(
                self.config[self.dataset]["video_frames_dir"],
                vname
            )
            frame = np.array(Image.open(image_file)) # Shape: (H, W, C)
            if len(frame.shape) == 2:
                frame = np.stack([frame] * 3, axis=-1)
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")
        return frame


class UDFProposer:
    def __init__(self, shared_resources: SharedResources):
        # Shared resources
        self.shared_resources = shared_resources
        self.config = shared_resources.config
        self.prompt_config = shared_resources.prompt_config
        self.dataset = shared_resources.dataset
        self.registered_functions = shared_resources.registered_functions
        self.object_domain = shared_resources.object_domain
        self.openai_model_name = shared_resources.openai_model_name
        self.run_id = shared_resources.run_id

        self.cost_estimation = defaultdict(float)

    def get_cost_estimation(self):
        return self.cost_estimation

    def propose(self, user_query):
        self.proposed_functions = {}
        # Step 1: propose new UDFs
        logger.info("Proposing new UDFs")
        if self.dataset in ["clevrer", "charades", "cityflow"]:  # video dataset
            dsl_definition_prompt = self.prompt_config["dsl_definition"]
        elif self.dataset in ["clevr", "gqa", "vaw"]:  # image dataset
            dsl_definition_prompt = self.prompt_config["dsl_definition_image"]
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")
        system_message = replace_slot(
            " ".join(
                [
                    dsl_definition_prompt,
                    self.prompt_config["udf_definition"]["without_object"] if self.dataset in ["clevr", "clevrer", "vaw", "cityflow"] else self.prompt_config["udf_definition"]["with_object"],
                    self.prompt_config["registered_udfs"],
                    self.prompt_config["propose_udfs"],
                ]
            ),
            {
                "functions": "\n".join(
                    [
                        "{}: {}".format(func["signature"], func["description"])
                        for func in self.registered_functions
                    ]
                )
            },
        )
        udf_proposer = autogen.AssistantAgent(
            name="udf_proposer",
            system_message=system_message,
            llm_config={
                "config_list": [{
                    'model': self.openai_model_name,
                    'api_key': os.getenv("OPENAI_API_KEY"),
                }],
                "timeout": 120,
                "temperature": self.config["udf_proposer"]["temperature"],
                "seed": self.run_id,
                "top_p": self.config["udf_proposer"]["top_p"],
                "max_tokens": 512,
                "cache_seed": None,
            },
        )

        # TODO: Add cost estimation
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "")
            and "terminate" in x.get("content", "").rstrip().lower(),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            # code_execution_config={"work_dir": "coding", "use_docker": False},
        )

        @user_proxy.register_for_execution()
        @udf_proposer.register_for_llm(
            description="Verify syntax correctness of proposed UDFs."
        )
        def verify_syntax(
            proposed_functions: Annotated[
                List[List[str]],
                "A list of proposed functions where proposed_functions[i] = [signature_i, description_i]. 'signature_i' represents the function signature 'function(args)', and 'description_i' contains the function description that starts with the word 'whether' and captures any specific definition as mentioned in the user query.",
            ]
        ) -> str:
            try:
                invalid_funcs = []
                for proposed_function in proposed_functions:
                    signature = proposed_function[0]
                    udf_name, udf_vars = parse_signature(signature)
                    if len(udf_vars) != 1 and len(udf_vars) != 2:
                        invalid_funcs.append(signature)
                if len(invalid_funcs) > 0:
                    return f"Invalid number of arguments for proposed functions: {invalid_funcs}."
                else:
                    for func in proposed_functions:
                        self.proposed_functions[func[0]] = func[1]
                    return "Success"
            except Exception as e:
                return "Error: " + str(e)

        logger.debug(f"system_message: {system_message}")

        user_proxy.initiate_chat(
            udf_proposer,
            message=f"User query: {user_query}",
        )
        usage_summary = gather_usage_summary([udf_proposer, user_proxy])["usage_including_cached_inference"][self.openai_model_name]
        self.cost_estimation["propose_udfs"] += usage_summary["prompt_tokens"] * MODEL_COST[self.openai_model_name][0] + usage_summary["completion_tokens"] * MODEL_COST[self.openai_model_name][1]

        try:
            logger.info(
                "Proposed functions: {}".format(self.proposed_functions)
            )  # key: signature, value: description
            registered_function_names = set(
                [
                    registered_function["signature"].split("(")[0].lower()
                    for registered_function in self.registered_functions
                ]
            )
            logger.info("filtering out functions that are already registered")
            for key in list(self.proposed_functions.keys()):
                if key.split("(")[0].lower() in registered_function_names:
                    logger.info(f"filtering out {key}")
                    del self.proposed_functions[key]
            # Step 2: verify functions (i.e., whether they can be constructed out of existing ones)
            # TODO: Implement this
            return self.proposed_functions
        except Exception as e:
            f"Error: {e}"
            return {}


class UDFGenerator(UtilsMixin):
    mlp_method = "three_clip"
    def __init__(self, shared_resources: SharedResources, udf_signature, udf_description, gt_udf_name):
        # Shared resources
        self.shared_resources = shared_resources
        self.config = shared_resources.config
        self.prompt_config = shared_resources.prompt_config
        self.registered_functions = shared_resources.registered_functions
        self.object_domain = shared_resources.object_domain
        self.relationship_domain = shared_resources.relationship_domain
        self.attribute_domain = shared_resources.attribute_domain
        self.dataset = shared_resources.dataset
        self.labeling_budget = shared_resources.labeling_budget
        self.n_selection_samples = shared_resources.n_selection_samples
        self.num_interpretations = shared_resources.num_interpretations
        self.num_parameter_search = shared_resources.num_parameter_search
        self.program_with_pixels = shared_resources.program_with_pixels
        self.program_with_pretrained_models = shared_resources.program_with_pretrained_models
        self.query_class_name = shared_resources.query_class_name
        self.query_id = shared_resources.query_id
        self.run_id = shared_resources.run_id
        self.num_workers = shared_resources.num_workers
        self.selection_strategy = shared_resources.selection_strategy
        self.selection_labels = shared_resources.selection_labels
        self.allow_kwargs_in_udf = shared_resources.allow_kwargs_in_udf
        self.llm_method = shared_resources.llm_method
        self.is_async = shared_resources.is_async
        self.openai_model_name = shared_resources.openai_model_name
        self.client = shared_resources.client
        self.executor = shared_resources.executor
        self.n_train = shared_resources.n_train
        self.n_test = shared_resources.n_test
        self.n_train_distill = shared_resources.n_train_distill
        self.n_test_distill = shared_resources.n_test_distill
        self.save_labeled_data = shared_resources.save_labeled_data
        self.load_labeled_data = shared_resources.load_labeled_data
        self.attribute_features_dir = shared_resources.attribute_features_dir
        self.relationship_features_dir = shared_resources.relationship_features_dir
        self.one_object_df = shared_resources.one_object_df
        self.two_objects_df = shared_resources.two_objects_df
        self.device = shared_resources.device
        self.clip_model = shared_resources.clip_model
        self.clip_processor = shared_resources.clip_processor
        self.tokenizer = shared_resources.tokenizer
        self.dim_in = shared_resources.dim_in
        self.llava_model = shared_resources.llava_model
        self.llava_processor = shared_resources.llava_processor
        self.vid_to_vname = shared_resources.vid_to_vname

        # Per-UDF state variables
        self.udf_signature = udf_signature
        self.udf_description = udf_description
        self.gt_udf_name = gt_udf_name
        self.udf_name, self.udf_vars = parse_signature(udf_signature)
        self.n_obj = len(self.udf_vars)
        self.llm_positive_df = None
        self.llm_negative_df = None
        self.cost_estimation = defaultdict(float)
        self.execution_time = defaultdict(float)
        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )


    def get_cost_estimation(self):
        return self.cost_estimation

    def get_execution_time(self):
        return self.execution_time

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), after=after_log(logger, logging.DEBUG))
    async def completion_with_backoff(self, **kwargs):
        return await self.client.chat.completions.create(**kwargs)


    async def implement(self):
        udf_candidate_list = await self._implement()
        return udf_candidate_list, self.llm_positive_df, self.llm_negative_df


    async def _implement(self):
        # TODO: incorporate labels (maybe in the selection stage)
        if self.selection_strategy == "program":
            if self.selection_labels == "none" and self.allow_kwargs_in_udf:
                self.allow_kwargs_in_udf = False
                logger.info(f"[{self.udf_signature}] Turning off allow_kwargs_in_udf since no labels are provided")
            return await self._generate_program()
        elif self.selection_strategy == "model":
            return await self._distill_model()
        elif self.selection_strategy == "llm":
            if self.selection_labels == "none" and self.allow_kwargs_in_udf:
                self.allow_kwargs_in_udf = False
                logger.info(f"[{self.udf_signature}] Turning off allow_kwargs_in_udf since no labels are provided")
            llm_decision = await self._llm_decides_udf_type()
            if llm_decision == "programUDF":
                return await self._generate_program()
            elif llm_decision == "modelUDF":
                return await self._distill_model()
            else:
                raise NotImplementedError(f"llm_decision: {llm_decision} is not supported yet.")
        elif self.selection_strategy == "both":
            program_udf_candidates = await self._generate_program()
            model_udf_candidates = await self._distill_model()
            return program_udf_candidates + model_udf_candidates


    async def _llm_decides_udf_type(self, udf_signature, udf_description):
        decide_udf_type_dict = self.prompt_config["decide_udf_type"]

        if self.program_with_pretrained_models:
            decide_udf_type_base_prompt = f'{decide_udf_type_dict["overall"]} {decide_udf_type_dict["code_with_pixels_and_pretrained_models"]} {decide_udf_type_dict["model"]} {decide_udf_type_dict["output"]}'
        elif self.program_with_pixels:
            decide_udf_type_base_prompt = f'{decide_udf_type_dict["overall"]} {decide_udf_type_dict["code_with_pixels"]} {decide_udf_type_dict["model"]} {decide_udf_type_dict["output"]}'
        else:
            decide_udf_type_base_prompt = f'{decide_udf_type_dict["overall"]} {decide_udf_type_dict["code_without_pixels"]} {decide_udf_type_dict["model"]} {decide_udf_type_dict["output"]}'

        decide_udf_type_prompt = replace_slot(
            decide_udf_type_base_prompt,
            {
                "udf_description": udf_description,
                "available_concepts": self.object_domain + self.relationship_domain + self.attribute_domain,
            },
        )
        logger.debug(f"[{self.udf_signature}] decide_udf_type_prompt: {decide_udf_type_prompt}")
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                logger.debug(f"[{self.udf_signature}] trial: {trial}")
                response = await self.completion_with_backoff(
                    model=self.openai_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert programming assistant.",
                        },
                        {"role": "user", "content": decide_udf_type_prompt},
                    ],
                    temperature=self.config["udf_generator"]["temperature"],
                    top_p=self.config["udf_generator"]["top_p"],
                    seed=self.run_id * 42 + trial,
                )
                llm_decision = response.choices[0].message.content
                self.cost_estimation["decide_udf_type"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
                logger.debug(f"[{self.udf_signature}] llm_decision: {llm_decision}")
                if "programUDF" in llm_decision:
                    return "programUDF"
                elif "modelUDF" in llm_decision:
                    return "modelUDF"
            except Exception as e:
                logger.exception(f"[{self.udf_signature}] ERROR: failed to decide UDF type: {e}")
                logger.debug(f"[{self.udf_signature}] {response}")


    ##############################################################
    #############                                    #############
    ############# UDF implementation (program-based) #############
    #############                                    #############
    ##############################################################
    async def _generate_program(self):
        """
        Implements the UDF program based on the given UDF signature and description.

        Args:
            udf_signature (str): The signature of the UDF.
            udf_description (str): The description of the UDF.

        Returns:
            list: A list of UDFCandidate objects representing the implemented UDFs.
        """
        _start = time.time()
        logger.info(f"[{self.udf_signature}] Program generation started")
        # Step 3: generate semantic interpretations and implement the UDF. Results are saved to disk
        generate_udfs_dict = self.prompt_config["generate_udfs"]
        attr_or_rel = "attribute" if self.n_obj == 1 else "relationship"
        if self.allow_kwargs_in_udf:
            if self.program_with_pretrained_models:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_with_optional_kwargs"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation_with_optional_kwargs"]}'
            elif self.program_with_pixels:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_with_optional_kwargs"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation_with_optional_kwargs"]}'
            else:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_with_optional_kwargs"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation_with_optional_kwargs"]}'
        else:
            if self.program_with_pretrained_models:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_pixels"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_pixels"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation"]}'
            elif self.program_with_pixels:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_pixels"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_pixels"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation"]}'
            else:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation"]}'
        # Construct python function arguments
        if self.n_obj == 1:
            py_func_args = [f"{self.udf_vars[0]}_oname", f"{self.udf_vars[0]}_x1", f"{self.udf_vars[0]}_y1", f"{self.udf_vars[0]}_x2", f"{self.udf_vars[0]}_y2", f"{self.udf_vars[0]}_anames", "height", "width"]
        else:
            py_func_args = [f"{self.udf_vars[0]}_oname", f"{self.udf_vars[0]}_x1", f"{self.udf_vars[0]}_y1", f"{self.udf_vars[0]}_x2", f"{self.udf_vars[0]}_y2", f"{self.udf_vars[0]}_anames", f"{self.udf_vars[1]}_oname", f"{self.udf_vars[1]}_x1", f"{self.udf_vars[1]}_y1", f"{self.udf_vars[1]}_x2", f"{self.udf_vars[1]}_y2", f"{self.udf_vars[1]}_anames", f"{self.udf_vars[0]}_{self.udf_vars[1]}_rnames", f"{self.udf_vars[1]}_{self.udf_vars[0]}_rnames", "height", "width"]
        if self.allow_kwargs_in_udf:
            py_func_args.append("**kwargs")
        if self.program_with_pixels:
            py_func_args.insert(0, "img")
        py_func_signature = f"{self.udf_name}({', '.join(py_func_args)})"
        logger.info(
            f"[{self.udf_signature}] Implementing UDF: {py_func_signature}, with {self.num_interpretations} semantic interpretations"
        )
        generate_udfs_prompt = replace_slot(
            generate_udfs_base_prompt,
            {
                "num_interpretations": self.num_interpretations,
                "udf_signature": py_func_signature,
                "udf_description": self.udf_description,
                "object_domain": self.object_domain,
                "relationship_domain": self.relationship_domain,
                "attribute_domain": self.attribute_domain,
                "o1": self.udf_vars[0],
                "o2": self.udf_vars[1] if self.n_obj == 2 else "",
                # "n_obj": "one object" if n_obj == 1 else "two objects",
            },
        )
        logger.debug(f"[{self.udf_signature}] generate_udfs_prompt: {generate_udfs_prompt}")
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                logger.debug(f"[{self.udf_signature}] trial: {trial}")
                self.execution_time["program_generation"] += time.time() - _start
                response = await self.completion_with_backoff(
                    model=self.openai_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert programming assistant.",
                        },
                        {"role": "user", "content": generate_udfs_prompt},
                    ],
                    temperature=self.config["udf_generator"]["temperature"],
                    top_p=self.config["udf_generator"]["top_p"],
                    seed=self.run_id * 42 + trial,
                )
                _start = time.time()
                self.cost_estimation["generate_program"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
                # NOTE: Sometimes GPT generates more UDFs than requested, so we remove the extra ones
                verifed_implemented_udfs = []
                implemented_udfs = eval(
                    "\n\n".join(
                        re.findall(
                            r"```json\n(.*?)```",
                            response.choices[0].message.content,
                            re.DOTALL,
                        )
                    )
                )["answer"][:self.num_interpretations]
                logger.debug(f"[{self.udf_signature}] implemented_udfs: {implemented_udfs}")
                for idx in range(len(implemented_udfs)):
                    implemented_udf = implemented_udfs[idx]
                    self.execution_time["program_generation"] += time.time() - _start
                    implemented_udf, success = await self.verify_syntax_correctness(implemented_udf, self.udf_vars, self.udf_name, py_func_signature, py_func_args, self.udf_description, self.n_obj, verify_syntax_correctness_base_prompt)
                    _start = time.time()
                    if success:
                        implemented_udf["udf_name"] = self.udf_name
                        implemented_udf["udf_signature"] = self.udf_signature
                        implemented_udf["udf_description"] = self.udf_description
                        verifed_implemented_udfs.append(implemented_udf)
                        logger.info(f"[{self.udf_signature}] [{idx}] semantic_interpretation: {implemented_udf['semantic_interpretation']}")
                        logger.info(f"[{self.udf_signature}] [{idx}] function_implementation: {implemented_udf['function_implementation']}")
                        logger.info(f"[{self.udf_signature}] [{idx}] kwargs: {implemented_udf.get('kwargs', {})}")
                break
            except Exception as e:
                logger.exception(f"[{self.udf_signature}] ERROR: failed to implement UDF: {e}")
                logger.debug(f"[{self.udf_signature}] {response}")

        # Read UDF candidates from json files
        udf_candidate_list = []  # List[UDFCandidate]
        for i in range(len(verifed_implemented_udfs)):
            try:
                udf_dict = verifed_implemented_udfs[i]
                if self.allow_kwargs_in_udf and udf_dict.get("kwargs", {}):
                    # Instantiate the kwargs with default values
                    udf_variant_dict = copy.deepcopy(udf_dict)
                    udf_variant_dict["kwargs"] = {k: v["default"] for k, v in udf_variant_dict["kwargs"].items() if v["default"] is not None}
                    new_udf_candidate = UDFCandidate(id=i, payload=udf_variant_dict)
                    logger.debug(f"[{self.udf_signature}] {new_udf_candidate}")
                    udf_candidate_list.append(new_udf_candidate)
                    # Instantiate the kwargs with values randomly sampled from the range
                    if self.num_parameter_search and self.num_parameter_search > 0:
                        for _ in range(self.num_parameter_search):
                            # deepcopy udf_dict
                            udf_variant_dict = copy.deepcopy(udf_dict)
                            for k in list(udf_variant_dict["kwargs"].keys()):
                                # randomly sample a value from the range
                                udf_variant_dict["kwargs"][k] = np.random.uniform(udf_variant_dict["kwargs"][k]["min"], udf_variant_dict["kwargs"][k]["max"])
                            new_udf_candidate = UDFCandidate(id=i, payload=udf_variant_dict)
                            logger.debug(f"[{self.udf_signature}] {new_udf_candidate}")
                            udf_candidate_list.append(new_udf_candidate)
                else: # No additional arguments
                    new_udf_candidate = UDFCandidate(id=i, payload=udf_dict)
                    logger.debug(f"[{self.udf_signature}] {new_udf_candidate}")
                    udf_candidate_list.append(new_udf_candidate)
            except Exception as e:
                logger.exception(f"[{self.udf_signature}] Failed to read UDF candidate: {e}")
        logger.info(f"[{self.udf_signature}] Program generation finished")
        self.execution_time["program_generation"] += time.time() - _start
        return udf_candidate_list

    async def verify_syntax_correctness(self, implemented_udf, udf_vars, udf_name, py_func_signature, py_func_args, udf_description, n_obj, verify_syntax_correctness_base_prompt, n_verify_samples=10):
        _start = time.time()
        df_samples = self.construct_train_and_test_data(n_obj, n_verify_samples, df_with_img_column=self.program_with_pixels)
        verify_syntax_correctness_prompt = replace_slot(
            verify_syntax_correctness_base_prompt,
            {
                "udf_signature": py_func_signature,
                "udf_description": udf_description,
                "semantic_interpretation": implemented_udf["semantic_interpretation"],
                "object_domain": self.object_domain,
                "relationship_domain": self.relationship_domain,
                "attribute_domain": self.attribute_domain,
                "o1": udf_vars[0],
                "o2": udf_vars[1] if n_obj == 2 else "",
            },
        )
        implemented_udf_json = json.dumps(implemented_udf)
        messages = [
            {"role": "user", "content": verify_syntax_correctness_prompt},
            {"role": "assistant", "content": "```json\n{}\n```".format(implemented_udf_json)}]
        success = False
        for retry in range(5):
            try:
                if retry != 0:
                    self.execution_time["program_generation"] += time.time() - _start
                    response = await self.completion_with_backoff(
                        model=self.openai_model_name,
                        messages=messages,
                        temperature=self.config["udf_generator"]["temperature"],
                        top_p=self.config["udf_generator"]["top_p"],
                        seed=self.run_id * 42,
                    )
                    _start = time.time()
                    self.cost_estimation["verify_syntax_correctness"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
                    messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    implemented_udf = eval(
                        "\n\n".join(
                            re.findall(
                                r"```json\n(.*?)```",
                                response.choices[0].message.content,
                                re.DOTALL,
                            )
                        )
                    )
                # Verify if the function has the correct number of arguments
                is_header_correct = True
                import_sklearn = False
                lines = implemented_udf["function_implementation"].split("\n")
                for _, line in enumerate(lines):
                    if line.startswith('def '):
                        generated_py_func_args = [arg.strip() for arg in line.split("(")[1].split(")")[0].split(",")]
                        if len(generated_py_func_args) != len(py_func_args):
                            messages.append({"role": "user", "content": f"Expected {len(py_func_args)} arguments, but got {len(generated_py_func_args)}. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
                            is_header_correct = False
                            break
                        for i, (gt_arg, generated_arg) in enumerate(zip(py_func_args, generated_py_func_args)):
                            if gt_arg != generated_arg:
                                messages.append({"role": "user", "content": f"Expected {gt_arg} as argument #{i+1}, but got {generated_arg}. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
                                is_header_correct = False
                                break
                    elif 'import' in line and 'sklearn' in line:
                        messages.append({"role": "user", "content": f"Using sklearn in multithreading environments may cause deadlock. Please do not use sklearn library. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
                        import_sklearn = True
                if (not is_header_correct) or import_sklearn:
                    continue
                py_func_name = "py_{}".format(udf_name)
                exec(implemented_udf["function_implementation"], globals())
                udf_obj = globals()[py_func_name]
                is_kwargs_correct = True
                kwargs = {}
                for k, v in implemented_udf.get("kwargs", {}).items():
                    if v["default"] is not None:
                        try:
                            kwargs[k] = float(v["default"])
                            v["min"] = float(v["min"])
                            v["max"] = float(v["max"])
                        except Exception as e:
                            messages.append({"role": "user", "content": f"Failed to parse kwargs due to the error: {type(e).__name__}: {e}. Please fix it and regenerate 'kwargs' using the same 'semantic_interpretation' and 'function_implementation'."})
                            is_kwargs_correct = False
                            break
                if not is_kwargs_correct:
                    continue
                result = self.exec_udf_with_data(df_samples, udf_obj, kwargs, n_obj, timeout=60)
                contains_non_boolean = False
                for r in result:
                    if r != 1 and r != 0:
                        contains_non_boolean = True
                        break
                if contains_non_boolean:
                    logger.debug(f"[{self.udf_signature}] The function returned non-boolean value: {result}")
                    messages.append({"role": "user", "content": f"The function returned non-boolean value, but it should return a boolean value. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
                    continue
                success = True
                break
            except Exception as e:
                messages.append({"role": "user", "content": f"Failed to execute the function due to the error: {type(e).__name__}: {e}. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
        if retry != 0:
            logger.debug(f"[{self.udf_signature}] verify_syntax_correctness:\n" + "\n".join([f"{message['role']}: {message['content']}" for message in messages]))
        self.execution_time["program_generation"] += time.time() - _start
        return implemented_udf, success


    ##############################################################
    #############                                    #############
    ############# UDF Distillation (distilled-model) #############
    #############                                    #############
    ##############################################################
    async def _distill_model(self):
        """
        gt_udf_name: ground truth UDF name. If provided, compute llm's TP, FP, TN, FN, f1 score and test the trained model.
        """
        logger.info(f"[{self.udf_signature}] Model distillation started")

        logger.info(f"[{self.udf_signature}] Model distillation (initialization) started")
        _start = time.time()
        attribute_df = self.conn.execute(f"SELECT * FROM {self.dataset}_attributes").df()
        relationship_df = self.conn.execute(f"SELECT * FROM {self.dataset}_relationships").df()

        # ask LLM about relevant object classes to the target relationships, and filter data
        filtered_objects, filtered_subjects, filtered_targets = None, None, None
        self.execution_time["model_distillation_init"] += time.time() - _start
        if self.dataset in ["charades"]:
            filtered_objects = list(set(await self.llm_filter_relevant_objects(self.udf_signature, self.udf_description) + ['person']))
        elif self.dataset in ["gqa", "vaw"]:
            if self.n_obj == 1:
                filtered_objects = await self.llm_filter_relevant_objects(self.udf_signature, self.udf_description)
            else: # n_obj == 2
                filtered_subjects, filtered_targets = await self.llm_filter_relevant_subjects_targets(self.udf_signature, self.udf_description)
        logger.debug(f"[{self.udf_signature}] filtered_objects: {filtered_objects}, filtered_subjects: {filtered_subjects}, filtered_targets: {filtered_targets}")
        logger.info(f"[{self.udf_signature}] Model distillation (initialization) finished")

        num_active_learning_rounds = (self.n_train_distill - 1) // 100
        labeled_indices = set()
        for active_learning_round in range(num_active_learning_rounds + 1):
            logger.info(f"[{self.udf_signature}] Active learning round: {active_learning_round}")

            if active_learning_round == 0:
                logger.info(f"[{self.udf_signature}] Model distillation (data loading) started")
                _start = time.time()
                # NOTE: LLM doesn't generate labels in some cases, so we need to double the number of samples (i.e., self.n_train * 1.2) to ensure we have enough training samples
                if self.gt_udf_name:
                    self.df_train, self.df_test = self.construct_train_and_test_data(self.n_obj, int(self.n_train_distill * 1.2), self.n_test_distill, df_with_img_column=True, filtered_objects=filtered_objects, filtered_subjects=filtered_subjects, filtered_targets=filtered_targets)
                else:
                    self.df_train = self.construct_train_and_test_data(self.n_obj, int(self.n_train_distill * 1.2), df_with_img_column=True, filtered_objects=filtered_objects, filtered_subjects=filtered_subjects, filtered_targets=filtered_targets)
                logger.info(f"[{self.udf_signature}] Model distillation (data loading) finished")
                self.execution_time["model_distillation_data_loading"] += time.time() - _start

            logger.info(f"[{self.udf_signature}] Model distillation (data labeling) started")
            await self.llm_annotate_data(active_learning_round=active_learning_round)
            logger.info(f"[{self.udf_signature}] Model distillation (data labeling) finished")

            logger.info(f"[{self.udf_signature}] Model distillation (model training) started")
            _start = time.time()
            self.mlp_prepare_data()
            best_ckpt = self.train(active_learning_round)
            if self.gt_udf_name and hasattr(self, 'df_test'):
                self.test()
            logger.info(f"[{self.udf_signature}] Model distillation (model training) finished")
            self.execution_time["model_distillation_model_training"] += time.time() - _start

            if active_learning_round < num_active_learning_rounds:
                logger.info(f"[{self.udf_signature}] Model distillation (active learning) started")
                _start = time.time()
                checkpoint = torch.load(best_ckpt)
                hyper_parameters = checkpoint["hyper_parameters"]
                best_mlp_model = mlp.MLPProd(**hyper_parameters)
                best_mlp_model.load_state_dict(checkpoint["state_dict"])
                best_mlp_model.eval()
                best_mlp_model.to(self.device)

                # Predict on the training split
                pred_dataset = PredImageDataset(self.conn, self.n_obj, self.attribute_features_dir, self.relationship_features_dir)
                pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=4096, num_workers=self.num_workers, shuffle=False)

                all_rows = []
                all_predictions = []
                all_uncertainties = []

                with torch.no_grad():
                    for row, feature in tqdm(pred_loader, file=sys.stdout):
                        feature = feature.to(self.device)
                        pred, uncertainty = best_mlp_model(feature)
                        all_rows.append(row)
                        all_predictions.append(pred.cpu())
                        all_uncertainties.append(uncertainty.cpu())

                rows = torch.cat(all_rows).tolist()
                predictions = torch.cat(all_predictions).tolist()
                uncertainties = torch.cat(all_uncertainties).tolist()

                if self.n_obj == 1:
                    check_df = pd.DataFrame(rows, columns=['vid', 'fid', 'oid'])
                    check_df['aname'] = self.gt_udf_name
                    result = check_df.merge(attribute_df, on=['vid', 'fid', 'oid', 'aname'], how='left', indicator=True)
                    result = result.drop_duplicates(subset=['vid', 'fid', 'oid', 'aname'])
                    result = result.rename(columns={"oid": "o1_oid"})
                else:
                    check_df = pd.DataFrame(rows, columns=['vid', 'fid', 'oid1', 'oid2'])
                    check_df['rname'] = self.gt_udf_name
                    result = check_df.merge(relationship_df, on=['vid', 'fid', 'oid1', 'oid2', 'rname'], how='left', indicator=True)
                    result = result.drop_duplicates(subset=['vid', 'fid', 'oid1', 'oid2', 'rname'])
                    result = result.rename(columns={"oid1": "o1_oid", "oid2": "o2_oid"})
                result['label'] = (result['_merge'] == 'both').astype(int)
                result = result.reset_index(drop=True)
                labels = result['label'].tolist()

                # Compute F1 score
                f1 = f1_score(labels, predictions)
                logger.info(f"[{self.udf_signature}] F1 score: {f1}")
                tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
                logger.info(f"[{self.udf_signature}] TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
                if self.dataset == "charades":
                    result['pred'] = predictions
                    result['uncertainty'] = uncertainties
                    result_human_object = result[result["o1_oid"] == 0]
                    labels_human_object = result_human_object["label"].tolist()
                    predictions_human_object = result_human_object["pred"].tolist()
                    f1_human_object = f1_score(labels_human_object, predictions_human_object)
                    logger.info(f"[{self.udf_signature}] [human-object only] F1 score: {f1_human_object}")
                    tn_1, fp_1, fn_1, tp_1 = confusion_matrix(labels_human_object, predictions_human_object).ravel()
                    logger.info(f"[{self.udf_signature}] [human-object only] TP: {tp_1}, FP: {fp_1}, TN: {tn_1}, FN: {fn_1}")

                # Active learning: select a batch of rows with the highest uncertainty that are not labeled
                selected_indices = np.argsort(-np.array(uncertainties))
                if self.dataset == "charades":
                    # "charades": only select the rows with the highest uncertainty for the human-object relationship
                    mask = (result['o1_oid'] == 0) & (result['vid'] < self.config[self.dataset]["dataset_size"] // 2)
                    filtered_indices = set(result.index[mask].tolist())
                    selected_indices = [i for i in selected_indices if i in filtered_indices and i not in labeled_indices]
                else:
                    # The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
                    mask = (result['vid'] < self.config[self.dataset]["dataset_size"] // 2)
                    filtered_indices = set(result.index[mask].tolist())
                    selected_indices = [i for i in selected_indices if i in filtered_indices and i not in labeled_indices]
                selected_indices = selected_indices[:min(100, self.n_train_distill - 100 * active_learning_round)]
                # Random sampling:
                # selected_indices = np.random.choice(len(uncertainties), min(100, self.n_train_distill - 100 * active_learning_round), replace=False)
                labeled_indices.update(selected_indices)
                logger.info(f"[{self.udf_signature}] labeled_indices: {sorted(labeled_indices)}, len(labeled_indices): {len(labeled_indices)}")
                if self.n_obj == 1:
                    columns = ['vid', 'fid', 'o1_oid']
                    df_source = self.one_object_df
                else:
                    columns = ['vid', 'fid', 'o1_oid', 'o2_oid']
                    df_source = self.two_objects_df
                selected_rows = result.iloc[selected_indices][columns]
                self.df_train = df_source.merge(selected_rows, on=columns, how='inner').reset_index(drop=True)
                self.df_train = self.df_train.drop_duplicates(subset=columns)
                self.df_train["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, self.df_train["vid"], self.df_train["fid"]), total=len(self.df_train), file=sys.stdout, desc="Processing frames"))
                logger.info(f"[{self.udf_signature}] Model distillation (active learning) finished")
                self.execution_time["model_distillation_active_learning"] += time.time() - _start

        udf_dict = {}
        udf_dict["udf_name"] = self.udf_name
        udf_dict["udf_signature"] = self.udf_signature
        udf_dict["udf_description"] = self.udf_description
        udf_dict["semantic_interpretation"] = 'model'
        udf_dict["function_implementation"] = best_ckpt
        new_udf_candidate = UDFCandidate(id='model', payload=udf_dict)
        logger.info(f"[{self.udf_signature}] Model distillation completed")
        return [new_udf_candidate]

    async def llm_filter_relevant_subjects_targets(self, udf_signature, udf_description):
        res = await self._llm_filter_relevant_training_data(udf_signature, udf_description, prompt_key="filter_subject_target")
        return res["subjects"], res["targets"]

    async def llm_filter_relevant_objects(self, udf_signature, udf_description):
        res = await self._llm_filter_relevant_training_data(udf_signature, udf_description, prompt_key="filter_object")
        return res["answer"]

    async def _llm_filter_relevant_training_data(self, udf_signature, udf_description, prompt_key):
        filter_objects_prompt = replace_slot(
            self.prompt_config[prompt_key],
            {
                "object_classes": self.object_domain,
                "udf_signature": udf_signature,
                "udf_description": udf_description,
            },
        )
        logger.debug(f"[{self.udf_signature}] filter_objects_prompt: {filter_objects_prompt}")
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                logger.debug(f"[{self.udf_signature}] trial: {trial}")
                response = await self.completion_with_backoff(
                    model=self.openai_model_name,
                    messages=[
                        {"role": "user", "content": filter_objects_prompt},
                    ],
                    temperature=self.config["udf_generator"]["temperature"],
                    top_p=self.config["udf_generator"]["top_p"],
                    seed=self.run_id * 42 + trial,
                )
                self.cost_estimation["filter_relevant_objects"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
                res = eval(
                    "\n\n".join(
                        re.findall(
                            r"```json\n(.*?)```",
                            response.choices[0].message.content,
                            re.DOTALL,
                        )
                    )
                )
                return res
            except Exception as e:
                logger.exception(f"[{self.udf_signature}] ERROR: failed to filter relevant objects: {e}")
                logger.debug(f"[{self.udf_signature}] {response}")

    async def llm_annotate_data(self, batch_size=8, active_learning_round=0):
        _start = time.time()
        labeled_data = defaultdict(list) # dictionary with 'train' and 'test' fields. Each field is a list of tuples (image_features, label)
        llm_positive_df = []
        llm_negative_df = []
        labeled_data_dir = os.path.join(self.config["model_dir"], "labeled_data", self.dataset, self.llm_method, self.query_class_name)
        labeled_data_path = os.path.join(labeled_data_dir, "udf-{}_query-{}_run-{}_ntrain-{}_labeled_data.pt".format(self.udf_name.lower(), self.query_id, self.run_id, self.n_train_distill))
        os.makedirs(labeled_data_dir, exist_ok=True)
        if self.load_labeled_data and os.path.exists(labeled_data_path):
            logger.info(f"[{self.udf_signature}] Loading labeled data from {labeled_data_path}")
            labeled_data = torch.load(labeled_data_path)
            for data in labeled_data['train']:
                logger.debug("[{}] row: {}".format(self.udf_signature, data['row'].drop('img').to_dict()))
                logger.debug("[{}] base64_image: {}".format(self.udf_signature, data["base64_image"]))
                logger.debug("[{}] gt_label: {}, llm_label: {}".format(self.udf_signature, data["label"], data["llm_label"]))
                if data["llm_label"] == 1:
                    llm_positive_df.append(data['row'])
                else:
                    llm_negative_df.append(data['row'])
            logger.debug("[{}] llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(self.udf_signature, labeled_data["metadata"]["llm_TP"], labeled_data["metadata"]["llm_FP"], labeled_data["metadata"]["llm_TN"], labeled_data["metadata"]["llm_FN"], labeled_data["metadata"]["llm_f1"]))
            if "test" in labeled_data:
                logger.debug("[{}] test_pos: {}, test_neg: {}".format(self.udf_signature, labeled_data["metadata"]["test_pos"], labeled_data["metadata"]["test_neg"]))
        else:
            self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN = 0, 0, 0, 0
            # Training and validation data
            self.label_count = 0
            if self.llm_method == "gpt4v":
                if self.is_async:
                    self.execution_time["model_distillation_data_labeling"] += time.time() - _start
                    tasks = [asyncio.create_task(self.label_one(row, labeled_data, llm_positive_df, llm_negative_df, active_learning_round)) for _, row in self.df_train.iterrows()]
                    try:
                        await asyncio.gather(*tasks)
                    except asyncio.CancelledError:
                        # Raised when we collected enough valid results. Cancel all other tasks
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                    _start = time.time()
                else:
                    for _, row in self.df_train.iterrows():
                        logger.debug(f"[{self.udf_signature}] +++++++++++++++++++++++++++++++++++++++++++++++")
                        try:
                            gt_label = self._get_gt_label(row)
                            # Read and crop frame
                            logger.debug("[{}] row: {}".format(self.udf_signature, row.drop('img').to_dict()))
                            frame, image_size = self.frame_processing_for_model(row)
                            if frame is None:
                                continue
                            self.execution_time["model_distillation_data_labeling"] += time.time() - _start
                            llm_label, base64_image, image_prompt = await self._llm_annotate_frame(frame, image_size, row, gt_label)
                            _start = time.time()
                            labeled_data['train'].append({"label": gt_label, "llm_label": llm_label, "base64_image": base64_image, "image_prompt": image_prompt, "row": row})
                            if llm_label == 1:
                                llm_positive_df.append(row)
                            else:
                                llm_negative_df.append(row)
                            if self.label_count >= min(100, self.n_train_distill - 100 * active_learning_round):
                                break
                        except Exception as e:
                            logger.exception(f"[{self.udf_signature}] Error: {e}")
                            continue
            elif self.llm_method == "llava":
                batched_rows = []
                batched_frames = []
                batched_image_sizes = []
                batched_gt_labels = []
                for _, row in self.df_train.iterrows():
                    try:
                        # Read and crop frame
                        frame, image_size = self.frame_processing_for_model(row)
                        if frame is None:
                            continue
                        gt_label = self._get_gt_label(row)
                        batched_rows.append(row)
                        batched_frames.append(frame)
                        batched_image_sizes.append(image_size)
                        batched_gt_labels.append(gt_label)
                        if len(batched_rows) == batch_size:
                            generated_text, llm_labels, base64_images, image_prompts = self._llava_annotate_frame(batched_frames, batched_image_sizes, batched_rows, batched_gt_labels)
                            for i in range(len(batched_rows)):
                                if llm_labels[i] == -1:
                                    continue
                                logger.debug(f"[{self.udf_signature}] +++++++++++++++++++++++++++++++++++++++++++++++")
                                logger.debug("[{}] row: {}".format(self.udf_signature, batched_rows[i].drop('img').to_dict()))
                                logger.debug("[{}] base64_image: {}".format(self.udf_signature, base64_images[i]))
                                logger.debug("[{}] Llava image prompt: {}".format(self.udf_signature, image_prompts[i]))
                                logger.debug("[{}] Llava result: {}".format(self.udf_signature, generated_text[i]))
                                logger.debug("[{}] gt_label: {}".format(self.udf_signature, batched_gt_labels[i]))
                                labeled_data['train'].append({"label": batched_gt_labels[i], "llm_label": llm_labels[i], "base64_image": base64_images[i], "image_prompt": image_prompts[i], "row": batched_rows[i]})
                                if llm_labels[i] == 1:
                                    llm_positive_df.append(batched_rows[i])
                                else:
                                    llm_negative_df.append(batched_rows[i])
                            if self.label_count >= self.n_train_distill:
                                break
                            batched_rows = []
                            batched_frames = []
                            batched_image_sizes = []
                            batched_gt_labels = []
                    except Exception as e:
                        logger.exception(f"[{self.udf_signature}] Error: {e}")
                        continue
            elif self.llm_method == "user": # Ground truth labels
                for _, row in self.df_train.iterrows():
                    logger.debug(f"[{self.udf_signature}] +++++++++++++++++++++++++++++++++++++++++++++++")
                    try:
                        gt_label = self._get_gt_label(row)
                        # Read and crop frame
                        logger.debug("[{}] row: {}".format(self.udf_signature, row.drop('img').to_dict()))
                        # frame, image_size = self.frame_processing_for_model(row)
                        # if self.n_obj == 2:
                        #     o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
                        #     cv2.rectangle(frame, (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
                        #     cv2.rectangle(frame, (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(255, 0, 0), thickness=1)
                        # _, buffer = cv2.imencode('.jpg', frame)
                        # base64_image = base64.b64encode(buffer).decode('utf-8')
                        # logger.debug("base64_image: {}".format(base64_image))
                        logger.debug("[{}] gt_label: {}".format(self.udf_signature, gt_label))
                        if gt_label == 1:
                            self.llm_TP += 1
                        else:
                            self.llm_TN += 1
                        self.label_count += 1
                        labeled_data['train'].append({"label": gt_label, "llm_label": gt_label, "base64_image": "", "image_prompt": "", "row": row})
                        if gt_label == 1:
                            llm_positive_df.append(row)
                        else:
                            llm_negative_df.append(row)
                        if self.label_count >= self.n_train_distill:
                            break
                    except Exception as e:
                        logger.exception(f"[{self.udf_signature}] Error: {e}")
                        continue
            if self.gt_udf_name is not None:
                llm_f1 = 2*self.llm_TP/(2*self.llm_TP+self.llm_FP+self.llm_FN) if 2*self.llm_TP+self.llm_FP+self.llm_FN > 0 else 0.0
                logger.debug("[{}] llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(self.udf_signature, self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN, llm_f1))
            else:
                llm_f1 = -1
            labeled_data["metadata"] = {"llm_TP": self.llm_TP, "llm_FP": self.llm_FP, "llm_TN": self.llm_TN, "llm_FN": self.llm_FN, "llm_f1": llm_f1}

            llm_positive_df = pd.DataFrame(llm_positive_df).reset_index(drop=True)
            self.llm_positive_df = pd.concat([self.llm_positive_df, llm_positive_df], ignore_index=True) if self.llm_positive_df is not None else llm_positive_df
            llm_negative_df = pd.DataFrame(llm_negative_df).reset_index(drop=True)
            self.llm_negative_df = pd.concat([self.llm_negative_df, llm_negative_df], ignore_index=True) if self.llm_negative_df is not None else llm_negative_df

            # Test data
            if active_learning_round == 0 and self.gt_udf_name is not None and hasattr(self, 'df_test'):
                for _, row in self.df_test.iterrows():
                    try:
                        gt_label = self._get_gt_label(row)
                        labeled_data['test'].append({"label": gt_label, "row": row})
                    except Exception as e:
                        logger.exception(f"[{self.udf_signature}] Error: {e}")
                        continue
                pos_count = sum([1 for data in labeled_data['test'] if data["label"] == 1])
                neg_count = sum([1 for data in labeled_data['test'] if data["label"] == 0])
                logger.debug(f"[{self.udf_signature}] test_pos: {pos_count}, test_neg: {neg_count}")
                labeled_data["metadata"]["test_pos"] = pos_count
                labeled_data["metadata"]["test_neg"] = neg_count
            # save labeled_data to a file
            if self.save_labeled_data:
                logger.info(f"[{self.udf_signature}] Saving labeled data to {labeled_data_path}")
                torch.save(labeled_data, labeled_data_path)
        if active_learning_round == 0:
            self.labeled_data = labeled_data
        else:
            self.labeled_data['train'].extend(labeled_data['train'])
        self.execution_time["model_distillation_data_labeling"] += time.time() - _start
        # if self.llm_method == "llava":
        #     del llava_model
        #     torch.cuda.empty_cache()

    def mlp_prepare_data(self):
        splits = ['train', 'test'] if self.gt_udf_name is not None else ['train']
        for split in splits:
            logger.info(f"[{self.udf_signature}] Processing {split} data")
            idx_to_remove = []
            for i in range(len(self.labeled_data[split])):
                if "image_features" in self.labeled_data[split][i]:
                    continue
                try:
                    data = self.labeled_data[split][i]
                    row = data['row']
                    frame, image_size = self.frame_processing_for_model(row)
                    if frame is None: # failed to read the frame
                        idx_to_remove.append(i)
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_features = self.extract_features(frame, row, image_size)
                    if self.dataset == "charades" and self.n_obj == 2:
                        # For charades, also include object class embeddings
                        text_features = self.extract_text_features(row)
                        self.labeled_data[split][i]["image_features"] = torch.cat([image_features, text_features], dim=-1)
                    else:
                        self.labeled_data[split][i]["image_features"] = image_features
                except Exception as e:
                    logger.exception(f"[{self.udf_signature}] Error: {e}")
                    idx_to_remove.append(i)
                    continue
            for i in reversed(idx_to_remove):
                del self.labeled_data[split][i]

        # use 20% of the training data as validation data
        self.train_dataset = CustomImageDataset(self.labeled_data['train'], train=True)
        if self.gt_udf_name is not None:
            test_dataset = CustomImageDataset(self.labeled_data['test'], train=False)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    def train(self, active_learning_round=-1):
        # logger.debug("mlp_config: {}".format(mlp_config))
        if self.dataset == "charades":
            mlp_dim_in = self.dim_in if self.n_obj == 1 else self.dim_in * 5
        else:
            mlp_dim_in = self.dim_in if self.n_obj == 1 else self.dim_in * 3
        logger.debug(f"[{self.udf_signature}] mlp_dim_in: {mlp_dim_in}") # should be 512 for clip-vit-base-patch32
        self.checkpoint_root = os.path.join(self.config["model_dir"], "model_udf", self.dataset, f"{self.llm_method}_{self.mlp_method}", self.udf_name.lower())
        if active_learning_round >= 0:
            self.checkpoint_root = os.path.join(self.checkpoint_root, f"active_learning_round_{active_learning_round}")

        # TODO: fine-tuning the learning rate
        # learningrate_callback = pl.callbacks.LearningRateFinder()
        # callbacks.append(learningrate_callback)

        best_model_score = float('inf')
        best_ckpt = None

        class_counts = [sum(data["llm_label"] == i for data in self.labeled_data['train']) for i in range(2)]
        try:
            self.class_weights = torch.tensor([1.0 / count for count in class_counts]).to(self.device)
        except ZeroDivisionError as e:
            logger.exception(f"[{self.udf_signature}] Error: {e}\nclass_counts: {class_counts}")
            self.class_weights = torch.tensor([1.0, 1.0]).to(self.device)

        logger.debug(f"[{self.udf_signature}] class_counts: {class_counts}, class_weights: {self.class_weights}")

        for i in range(10):
            logger.debug(f"[{self.udf_signature}] Training model: trial {i}")
            self.checkpoint_filename = "udf={}-run={}-ntrain={}-trial={}".format(self.udf_name.lower(), self.run_id, self.n_train_distill, i)
            os.makedirs(self.checkpoint_root, exist_ok=True)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filename=self.checkpoint_filename,
                monitor="val_loss",
                mode="min",
            )
            callbacks=[checkpoint_callback]
            earlystopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
            callbacks.append(earlystopping_callback)
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)

            train_set_size = int(len(self.train_dataset) * 0.8)
            valid_set_size = len(self.train_dataset) - train_set_size
            train_split, valid_split = torch.utils.data.random_split(self.train_dataset, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(self.run_id * 42 + i))

            self.train_loader = torch.utils.data.DataLoader(train_split, batch_size=512, shuffle=True)
            self.val_loader = torch.utils.data.DataLoader(valid_split, batch_size=512, shuffle=False)

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
            current_model_score = checkpoint_callback.best_model_score
            logger.debug(f"[{self.udf_signature}] current_model_score: {current_model_score}, best_model_score: {min(best_model_score, current_model_score)}")
            if current_model_score < best_model_score:
                best_model_score = current_model_score
                best_ckpt = checkpoint_callback.best_model_path
        logger.debug(f"[{self.udf_signature}] Best model checkpoint: {best_ckpt}")
        # best_mlp_model = mlp.MLP.load_from_checkpoint(best_ckpt)
        return best_ckpt

    def test(self):
        # Inference:
        # model = mlp.MLP.load_from_checkpoint(self.dim_in, 2)
        logger.debug(f"[{self.udf_signature}] test with last model: ")
        self.trainer.test(self.mlp_model, dataloaders=self.test_loader)
        logger.debug(f"[{self.udf_signature}] test with best model: ")
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
            factor = 1 if self.dataset == "cityflow" else 1.5
            x1, y1, x2, y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size, factor=factor)
            frame = frame[y1:y2, x1:x2]
        else:
            o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
            o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
            frame = frame[min(o1_y1, o2_y1):max(o1_y2, o2_y2), min(o1_x1, o2_x1):max(o1_x2, o2_x2)]
        # resize the frame to 224x224
        frame = cv2.resize(frame, (224, 224))
        return frame, image_size


    async def label_one(self, row, labeled_data, llm_positive_df, llm_negative_df, active_learning_round):
        log_msgs = []
        log_msgs.append(f"[{self.udf_signature}] +++++++++++++++++++++++++++++++++++++++++++++++")
        # logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++")

        try:
            gt_label = self._get_gt_label(row)
            # Read and crop frame
            log_msgs.append("[{}] row: {}".format(self.udf_signature, row.drop('img').to_dict()))
            # logger.debug("row: {}".format(row.drop('img').to_dict()))
            frame, image_size = self.frame_processing_for_model(row)
            if frame is None:
                logger.debug("\n".join(log_msgs))
                return
            llm_label, base64_image, image_prompt = await self._async_llm_annotate_frame(frame, image_size, row, gt_label, log_msgs)
        except Exception as e:
            logger.debug("\n".join(log_msgs))
            logger.exception(f"[{self.udf_signature}] Error: {e}")
            return

        if len(labeled_data['train']) >= min(100, self.n_train_distill - 100 * active_learning_round):
            raise asyncio.CancelledError

        logger.debug("\n".join(log_msgs))
        labeled_data['train'].append({"label": gt_label, "llm_label": llm_label, "base64_image": base64_image, "image_prompt": image_prompt, "row": row})
        if llm_label == 1:
            llm_positive_df.append(row)
        else:
            llm_negative_df.append(row)


    async def _async_llm_annotate_frame(self, frame, image_size, row, gt_label, log_msgs):
        # TODO: don't resize the frame here, resize it in the model
        # Convert the frame to a base 64 encoded image
        # TODO: Try different llm annotation prompt: draw bounding boxes on subject and object.
        if self.n_obj == 2:
            o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
            cv2.rectangle(frame, (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
            cv2.rectangle(frame, (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(255, 0, 0), thickness=1)
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        log_msgs.append(f"[{self.udf_signature}] base64_image: {base64_image}")
        # logger.debug("base64_image: {}".format(base64_image))
        image_prompt = self._create_image_prompt(row, image_size)
        log_msgs.append(f"[{self.udf_signature}] Image prompt: {image_prompt}")
        # logger.debug("Image prompt: {}".format(image_prompt))
        response = await self.completion_with_backoff(
            model=self.openai_model_name,
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
        self.cost_estimation["model_udf_data_labeling"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
        log_msgs.append(f"[{self.udf_signature}] Result: {result}")
        # logger.debug("Result: {}".format(result))
        log_msgs.append(f"[{self.udf_signature}] gt_label: {gt_label}")
        # logger.debug("gt_label: {}".format(gt_label))
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

    async def _llm_annotate_frame(self, frame, image_size, row, gt_label):
        # TODO: don't resize the frame here, resize it in the model
        # Convert the frame to a base 64 encoded image
        # TODO: Try different llm annotation prompt: draw bounding boxes on subject and object.
        _start = time.time()
        if self.n_obj == 2:
            o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
            cv2.rectangle(frame, (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
            cv2.rectangle(frame, (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(255, 0, 0), thickness=1)
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        logger.debug(f"[{self.udf_signature}] base64_image: {base64_image}")
        image_prompt = self._create_image_prompt(row, image_size)
        logger.debug(f"[{self.udf_signature}] Image prompt: {image_prompt}")
        self.execution_time["model_distillation_data_labeling"] += time.time() - _start
        response = await self.completion_with_backoff(
            model=self.openai_model_name,
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
        _start = time.time()
        result = response.choices[0].message.content
        self.cost_estimation["model_udf_data_labeling"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
        logger.debug(f"[{self.udf_signature}] Result: {result}")
        logger.debug(f"[{self.udf_signature}] gt_label: {gt_label}")
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
            self.execution_time["model_distillation_data_labeling"] += time.time() - _start
            raise ValueError("Invalid response", result)
        self.label_count += 1
        self.execution_time["model_distillation_data_labeling"] += time.time() - _start
        return llm_label, base64_image, image_prompt

    def _create_image_prompt(self, row, image_size):
        image_prompt = "{}? Answer with 'yes' or 'no'.".format(self.replace_objects(self.udf_description.rstrip(string.punctuation), row, image_size))
        return image_prompt

    def _llava_annotate_frame(self, batched_frames, batched_image_sizes, batched_rows, batched_gt_labels):
        llm_labels, base64_images, llava_image_prompts = [], [], []
        pil_images = []
        for i in range(len(batched_frames)):
            if self.n_obj == 2:
                # Draw bounding boxes on subject and object
                o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(batched_rows[i], batched_image_sizes[i])
                cv2.rectangle(batched_frames[i], (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
                cv2.rectangle(batched_frames[i], (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(255, 0, 0), thickness=1)
            _, buffer = cv2.imencode('.jpg', batched_frames[i])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            base64_images.append(base64_image)
            batched_frames[i] = cv2.cvtColor(batched_frames[i], cv2.COLOR_BGR2RGB)
            # Convert the frame to PIL image
            pil_image = Image.fromarray(batched_frames[i])
            pil_images.append(pil_image)
            llava_image_prompt, last_word = self._create_llava_image_prompt(batched_rows[i], batched_image_sizes[i])
            llava_image_prompts.append(llava_image_prompt)
        inputs = self.llava_processor(llava_image_prompts, pil_images, padding=True, return_tensors="pt").to(self.device)
        output = self.llava_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.2,
            top_p=0.7
        )
        generated_text = self.llava_processor.batch_decode(output, skip_special_tokens=True)
        results = []
        for i, result in enumerate(generated_text):
            result = result.split(last_word)[-1].strip().lower()
            results.append(result)
            if "yes" in result.lower():
                llm_label = 1
                if self.gt_udf_name is not None:
                    if batched_gt_labels[i] == 1:
                        self.llm_TP += 1
                    else:
                        self.llm_FP += 1
                self.label_count += 1
            elif "no" in result.lower():
                llm_label = 0
                if self.gt_udf_name is not None:
                    if batched_gt_labels[i] == 0:
                        self.llm_TN += 1
                    else:
                        self.llm_FN += 1
                self.label_count += 1
            else:
                llm_label = -1
            llm_labels.append(llm_label)
        return results, llm_labels, base64_images, llava_image_prompts

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

    def extract_text_features(self, row):
        if self.n_obj == 1:
            text = row["o1_oname"]
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**inputs)
            outputs = outputs.squeeze(0)
        else:
            inputs = self.tokenizer([row["o1_oname"], row["o2_oname"]], padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**inputs) # 2 x 512
            outputs = outputs.reshape(-1)

        return outputs


class UDFSelector(UtilsMixin):
    def __init__(self, shared_resources: SharedResources, llm_positive_df, llm_negative_df):
        # Shared resources
        self.shared_resources = shared_resources
        self.config = shared_resources.config
        self.prompt_config = shared_resources.prompt_config
        self.registered_functions = shared_resources.registered_functions
        self.object_domain = shared_resources.object_domain
        self.relationship_domain = shared_resources.relationship_domain
        self.attribute_domain = shared_resources.attribute_domain
        self.dataset = shared_resources.dataset
        self.labeling_budget = shared_resources.labeling_budget
        self.n_selection_samples = shared_resources.n_selection_samples
        self.num_interpretations = shared_resources.num_interpretations
        self.num_parameter_search = shared_resources.num_parameter_search
        self.program_with_pixels = shared_resources.program_with_pixels
        self.program_with_pretrained_models = shared_resources.program_with_pretrained_models
        self.query_class_name = shared_resources.query_class_name
        self.query_id = shared_resources.query_id
        self.run_id = shared_resources.run_id
        self.num_workers = shared_resources.num_workers
        self.selection_strategy = shared_resources.selection_strategy
        self.selection_labels = shared_resources.selection_labels
        self.allow_kwargs_in_udf = shared_resources.allow_kwargs_in_udf
        self.llm_method = shared_resources.llm_method
        self.is_async = shared_resources.is_async
        self.openai_model_name = shared_resources.openai_model_name
        self.executor = shared_resources.executor
        self.n_train = shared_resources.n_train
        self.n_test = shared_resources.n_test
        self.n_train_distill = shared_resources.n_train_distill
        self.n_test_distill = shared_resources.n_test_distill
        self.save_labeled_data = shared_resources.save_labeled_data
        self.load_labeled_data = shared_resources.load_labeled_data
        self.attribute_features_dir = shared_resources.attribute_features_dir
        self.relationship_features_dir = shared_resources.relationship_features_dir
        self.one_object_df = shared_resources.one_object_df
        self.two_objects_df = shared_resources.two_objects_df
        self.device = shared_resources.device
        self.clip_model = shared_resources.clip_model
        self.clip_processor = shared_resources.clip_processor
        self.tokenizer = shared_resources.tokenizer
        self.dim_in = shared_resources.dim_in
        self.llava_model = shared_resources.llava_model
        self.llava_processor = shared_resources.llava_processor
        self.vid_to_vname = shared_resources.vid_to_vname

        # Per-UDF state variables
        self.llm_positive_df = llm_positive_df
        self.llm_negative_df = llm_negative_df
        self.cost_estimation = defaultdict(float)
        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )


    def get_cost_estimation(self):
        return self.cost_estimation


    ###########################
    ######               ######
    ###### UDF Selection ######
    ######               ######
    ###########################
    # TODO: If all the generated UDFs are terrible, we do rather not select any UDF (by using a dummy UDF that always returns True).
    # We could either add a dummy UDF to the candidate list,
    # or only register UDFs when the F1 score is above a certain threshold (how to decide the threshold?) to avoid generating terrible UDFs
    def select(self, gt_udf_name, udf_candidate_list):
        df_with_img_column = self.program_with_pixels
        for udf_candidate in udf_candidate_list:
            if udf_candidate.id == "model":
                df_with_img_column = True
                break
        return self._select(gt_udf_name, udf_candidate_list, df_with_img_column=df_with_img_column)


    def _select(self, gt_udf_name, udf_candidate_list, df_with_img_column):
        if len(udf_candidate_list) == 0:
            return None
        # if len(udf_candidate_list) == 1:
        #     selected_udf_candidate = udf_candidate_list[0]
        # else:
        udf_signature = udf_candidate_list[0].udf_signature
        udf_name, udf_vars = parse_signature(udf_signature)
        n_obj = len(udf_vars)
        # Add a dummy UDF that always returns True
        udf_description = udf_candidate_list[0].udf_description
        dummy_udf = UDFCandidate(id='dummy', payload={'udf_name': udf_name, 'udf_signature': udf_signature, 'udf_description': udf_description, 'semantic_interpretation': 'dummy', 'function_implementation': f'def py_{udf_name}(*args, **kwargs):\n    return True\n'})
        udf_candidate_list.append(dummy_udf)

        # Construct training data and test data
        df_train, df_test = self.construct_train_and_test_data(n_obj, self.n_train, self.n_test, df_with_img_column=df_with_img_column)

        # Select new video segments to label
        # TODO: df_train and llm_positive_df may contain the same tuples
        labeled_index = []
        llm_positive_labeled_index = []
        llm_negative_labeled_index = []
        segment_selection_time = 0
        _start_segment_selection_time = time.time()
        # TODO: perhaps regenerate one more UDF based on current labels after every k iterations

        sampling_strategy = SamplingStrategy.positive
        for iter in range(self.labeling_budget):
            logger.info("iter {}: {}".format(iter, sampling_strategy))
            _start_segment_selection_time_per_iter = time.time()

            if sampling_strategy == SamplingStrategy.positive and self.llm_positive_df is not None and len(self.llm_positive_df) > len(llm_positive_labeled_index):
                new_labeled_index = [len(llm_positive_labeled_index)]
                logger.info("pick next segments from llm_positive_df {}".format(new_labeled_index))
                llm_positive_labeled_index += new_labeled_index
                new_labeled_df = self.llm_positive_df.iloc[new_labeled_index]
            elif sampling_strategy == SamplingStrategy.negative and self.llm_negative_df is not None and len(self.llm_negative_df) > len(llm_negative_labeled_index):
                new_labeled_index = [len(llm_negative_labeled_index)]
                logger.info("pick next segments from llm_negative_df {}".format(new_labeled_index))
                llm_negative_labeled_index += new_labeled_index
                new_labeled_df = self.llm_negative_df.iloc[new_labeled_index]
            else:
                new_labeled_index = self.select_sample(
                    udf_candidate_list, udf_name, df_train, n_obj, labeled_index, sampling_strategy
                )
                logger.info("pick next segments {}".format(new_labeled_index))
                labeled_index += new_labeled_index
                new_labeled_df = df_train.iloc[new_labeled_index]
            logger.info("# labeled segments {}".format(len(set(llm_positive_labeled_index)) + len(set(llm_negative_labeled_index)) + len(set(labeled_index))))
            if n_obj == 1:
                labeled_df_list = [df_train.iloc[labeled_index]['o1_gt_anames']]
                if self.llm_positive_df is not None and len(self.llm_positive_df):
                    labeled_df_list.append(self.llm_positive_df.iloc[llm_positive_labeled_index]['o1_gt_anames'])
                if self.llm_negative_df is not None and len(self.llm_negative_df):
                    labeled_df_list.append(self.llm_negative_df.iloc[llm_negative_labeled_index]['o1_gt_anames'])
                y_true = pd.Series([gt_udf_name in anames for anames in pd.concat(labeled_df_list)])
            elif n_obj == 2:
                labeled_df_list = [df_train.iloc[labeled_index]['o1_o2_gt_rnames']]
                if self.llm_positive_df is not None and len(self.llm_positive_df):
                    labeled_df_list.append(self.llm_positive_df.iloc[llm_positive_labeled_index]['o1_o2_gt_rnames'])
                if self.llm_negative_df is not None and len(self.llm_negative_df):
                    labeled_df_list.append(self.llm_negative_df.iloc[llm_negative_labeled_index]['o1_o2_gt_rnames'])
                logger.debug(f"pd.concat(labeled_df_list): {pd.concat(labeled_df_list)}")
                y_true = pd.Series([gt_udf_name in rnames for rnames in pd.concat(labeled_df_list)])
            # log number of positive and negative samples
            logger.info(
                "# positive: {}, # negative: {}".format(
                    sum(y_true), len(y_true) - sum(y_true)
                )
            )

            # Decide sampling strategy of next iteration
            if sum(y_true) <= len(y_true) - sum(y_true):
                sampling_strategy = SamplingStrategy.positive
            elif len(y_true) - sum(y_true) < 3:
                sampling_strategy = SamplingStrategy.negative
            else:
                sampling_strategy = SamplingStrategy.uncertainty

            # Update scores
            indices_to_remove = []
            for i in range(len(udf_candidate_list)):
                labeled_df = [df_train.iloc[labeled_index]]
                if self.llm_positive_df is not None and len(self.llm_positive_df):
                    labeled_df.append(self.llm_positive_df.iloc[llm_positive_labeled_index])
                if self.llm_negative_df is not None and len(self.llm_negative_df):
                    labeled_df.append(self.llm_negative_df.iloc[llm_negative_labeled_index])
                labeled_df = pd.concat(labeled_df)
                try:
                    score, loss_t = self.compute_udf_score(
                        gt_udf_name,
                        udf_candidate_list[i],
                        udf_name,
                        n_obj,
                        labeled_df,
                        new_labeled_df,
                        add_one=True, # add one to avoid zero f1 score
                    )
                    udf_candidate_list[i].score = score
                    udf_candidate_list[i].loss_t += loss_t
                except Exception as e:
                    logger.exception(f"ERROR: failed to execute UDFCandidate(id={udf_candidate_list[i].id}): {e}")
                    indices_to_remove.append(i)
                    continue
            # Remove UDFs that failed to execute
            for i in sorted(indices_to_remove, reverse=True):
                del udf_candidate_list[i]
            # sort udf_candidate_list by score
            udf_candidate_list_sorted = sorted(
                udf_candidate_list, key=lambda x: x.score, reverse=True
            )
            logger.debug("updated udf_candidate_list: {}".format("\n".join([str(e) for e in udf_candidate_list_sorted])))
            logger.debug(
                "test segment_selection_time_per_iter time: {}".format(
                    time.time() - _start_segment_selection_time_per_iter
                )
            )
        segment_selection_time += time.time() - _start_segment_selection_time
        logger.debug(
            "test segment_selection_time time: {}".format(segment_selection_time)
        )

        # compute test F1 score
        logger.info("compute test F1 score")
        for i in range(len(udf_candidate_list)):
            try:
                udf_candidate_list[i].test_score = self.compute_udf_score(
                    gt_udf_name,
                    udf_candidate_list[i],
                    udf_name,
                    n_obj,
                    df_test,
                )
                logger.info(str(udf_candidate_list[i]))
            except Exception as e:
                logger.exception(f"ERROR: failed to compute test F1 score of UDFCandidate(id={udf_candidate_list[i].id}): {e}")
                udf_candidate_list[i].test_score = -1
                continue

        logger.info("compute train F1 score")
        # compute the F1 score of the best udf (median F1 scores if there are multiple udfs with the same best score on the training set) on the test dataset
        if sum(y_true) == 0:
            logger.info("No positive samples are labeled. Returning the dummy UDF.")
            selected_udf_candidate = [udf_candidate for udf_candidate in udf_candidate_list if udf_candidate.id == "dummy"][0]
        else:
            # Compute final f1 score (without adding one)
            for i in range(len(udf_candidate_list)):
                labeled_df = [df_train.iloc[labeled_index]]
                if self.llm_positive_df is not None and len(self.llm_positive_df):
                    labeled_df.append(self.llm_positive_df.iloc[llm_positive_labeled_index])
                if self.llm_negative_df is not None and len(self.llm_negative_df):
                    labeled_df.append(self.llm_negative_df.iloc[llm_negative_labeled_index])
                labeled_df = pd.concat(labeled_df)
                try:
                    udf_candidate_list[i].score = self.compute_udf_score(
                        gt_udf_name,
                        udf_candidate_list[i],
                        udf_name,
                        n_obj,
                        labeled_df,
                    )
                except Exception as e:
                    logger.exception(f"ERROR: failed to compute final f1 score of UDFCandidate(id={udf_candidate_list[i].id}): {e}")
                    udf_candidate_list[i].score = -1
                    continue
            best_score = max(udf_candidate.score for udf_candidate in udf_candidate_list)
            best_candidates = [
                udf_candidate
                for udf_candidate in udf_candidate_list
                if udf_candidate.score == best_score
            ]

            f1_score_test_list = []
            for best_candidate in best_candidates:
                f1_score_test_list.append(best_candidate.test_score)
            median_f1_score_test = np.median(f1_score_test_list)
            logger.info("median test f1: {}".format(median_f1_score_test))
            # TODO: If there are multiple best udfs, select the one with faster execution time?
            # If there are multiple best udfs, dummy UDF will be preferred
            selected_udf_candidate = best_candidates[-1]

        if selected_udf_candidate.id not in ["model", "dummy"]:
            # Transforms the function by removing **kwargs from the function signature and replacing the kwargs with the actual values
            selected_udf_candidate.function_implementation = transform_function(
                original_code=selected_udf_candidate.function_implementation,
                instantiation_dict=selected_udf_candidate.kwargs,
            )
        logger.info(f"[Selected]: {str(selected_udf_candidate)}")
        return selected_udf_candidate


    def select_sample(
        self, udf_candidate_list, udf_name, df_train, n_obj, labeled_index, sampling_strategy
    ):


        prediction_matrix = []
        _start = time.time()

        # query_list = [query_graph.program for query_graph, _ in self.candidate_queries[:self.pool_size]]
        # logger.debug("query pool", [program_to_dsl(query, self.rewrite_variables) for query in query_list])
        unlabeled_index = np.setdiff1d(
            np.arange(len(df_train)), labeled_index, assume_unique=True
        )
        logger.debug("len(unlabeled_index): {}".format(len(unlabeled_index)))

        # sample a subset of videos during each iteration
        # If more than n_selection_samples videos, sample n_selection_samples videos
        if len(unlabeled_index) > self.n_selection_samples:
            sampled_index = np.random.choice(
                unlabeled_index, self.n_selection_samples, replace=False
            )
        else:
            sampled_index = unlabeled_index

        df_sampled = df_train.iloc[sampled_index]

        indices_to_remove = []
        for i, udf_candidate in enumerate(udf_candidate_list):
            logger.debug(f"Running udf_candidate(id={udf_candidate.id}) on sampled data")
            if udf_candidate.id == "model":
                # distilled-model UDF
                best_ckpt = udf_candidate.function_implementation
                predictions = self.predict_with_data(df_sampled, best_ckpt, n_obj)
                logger.debug("predictions: {}".format(predictions))
                prediction_matrix.append(predictions)
            else:
                try:
                    # program-based UDF
                    # For each sampled row in df_train, construct o1 and o2
                    kwargs = {}
                    for k, v in udf_candidate.kwargs.items():
                        kwargs[k] = float(v)
                    py_func_name = "py_{}".format(udf_name)
                    exec(udf_candidate.function_implementation, globals())
                    udf_obj = globals()[py_func_name]
                    # TODO: may need to timeout if running for too long
                    logger.debug(f"udf_name: {udf_name}")
                    result = self.exec_udf_with_data(df_sampled, udf_obj, kwargs, n_obj, requires_no_error=False, timeout=max(60, len(df_sampled)*0.2))
                    contains_non_boolean = False
                    for r in result:
                        if r != 1 and r != 0:
                            contains_non_boolean = True
                            break
                    if contains_non_boolean:
                        logger.debug(
                            f"ERROR: UDFCandidate(id={udf_candidate.id}) returned non-boolean value: {result}"
                        )
                        indices_to_remove.append(i)
                        continue
                except Exception as e:
                    logger.exception(f"ERROR: failed to execute UDFCandidate(id={udf_candidate.id}): {e}")
                    indices_to_remove.append(i)
                    continue
                prediction_matrix.append(result)
        # Remove UDFs that failed to execute
        for i in sorted(indices_to_remove, reverse=True):
            del udf_candidate_list[i]

        prediction_matrix = np.array(
            prediction_matrix
        ).transpose()  # (n_samples, n_udfs)
        logger.debug(
            "constructing prediction matrix took {} seconds".format(
                time.time() - _start
            )
        )
        logger.debug("prediction_matrix size {}".format(prediction_matrix.shape))

        # Reference: Learning Rare Category Classifiers on a Tight Labeling Budget
        # If the number of positives labeled so far is less than the number of negatives, we ask the human to label the most-likely positive images, otherwise we ask the human to label the images closest to the linear model’s margin
        eta_0 = np.sqrt(np.log(len(udf_candidate_list)) / 2)

        # Use F1-scores as weights
        posterior_t = [udf_candidate.score for udf_candidate in udf_candidate_list]
        # Use the original weights as in the paper
        # eta = eta_0 / np.sqrt(n_selection_samples)
        # loss_t = [loss_t for _, _, _, loss_t in udf_candidates_with_scores]
        # posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))

        posterior_t /= np.sum(posterior_t)  # normalized weight

        logger.debug("query weights {}".format(posterior_t))

        if sampling_strategy == SamplingStrategy.positive:
            # TODO: filter objects?
            # TODO: ask LLM?
            # find sample with highest weighted probability of being positive
            probability_list = np.zeros(len(sampled_index))
            for i in range(len(sampled_index)):
                probability_list[i] = np.inner(posterior_t, prediction_matrix[i, :])
            ind = np.argsort(-probability_list)
            logger.debug("probability list (desc): {}".format(probability_list[ind]))
            logger.debug("sampled index {}".format(sampled_index[ind]))
            max_probability_index = sampled_index[np.argmax(probability_list)]
            return [max_probability_index]
        elif sampling_strategy == SamplingStrategy.negative:
            probability_list = np.zeros(len(sampled_index))
            for i in range(len(sampled_index)):
                probability_list[i] = np.inner(posterior_t, prediction_matrix[i, :])
            ind = np.argsort(probability_list)
            logger.debug("probability list (asc): {}".format(probability_list[ind]))
            logger.debug("sampled index {}".format(sampled_index[ind]))
            min_probability_index = sampled_index[np.argmin(probability_list)]
            return [min_probability_index]
        else:
            entropy_list = np.zeros(len(sampled_index))
            for i in range(len(sampled_index)):
                entropy_list[i] = self._compute_u_t(posterior_t, prediction_matrix[i, :])
            ind = np.argsort(-entropy_list)
            logger.debug("entropy list {}".format(entropy_list[ind]))
            # df_object_pairs_train[sampled_index[ind]].apply(lambda row: logger.info("o1: {}, o2: {}".format(row["o1"], row["o2"])), axis=1)
            logger.debug("sampled index {}".format(sampled_index[ind]))
            # find argmax of entropy (top k)
            max_entropy_index = sampled_index[np.argmax(entropy_list)]
            return [max_entropy_index]

    def _compute_u_t(self, posterior_t, predictions_c):
        # Initialize possible u_t's
        u_t_list = np.zeros(2)

        # Repeat for each class
        for c in [0, 1]:
            # Compute the loss of models if the label of the streamed data is "c"
            loss_c = np.array(predictions_c != c) * 1
            # Compute the respective u_t value (conditioned on class c)
            term1 = np.inner(posterior_t, loss_c)
            u_t_list[c] = term1 * (1 - term1)

        # Return the final u_t
        u_t = np.max(u_t_list)

        return u_t

    def predict_with_data(self, df, ckpt, n_obj):
        # Predict the labels of all the data points
        checkpoint = torch.load(ckpt)
        hyper_parameters = checkpoint["hyper_parameters"]
        mlp_model = mlp.MLPProd(**hyper_parameters)
        mlp_model.load_state_dict(checkpoint["state_dict"])
        mlp_model.eval()
        mlp_model.to(self.device)

        # extract image features
        transforms = T.Compose([
            # T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            # T.CenterCrop(224),
            T.Lambda(lambda x: x / 255.0),        # Scale image data to [0, 1]
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        batch_size = 256
        if n_obj == 1:
            features, idxs_to_predict = self.extract_features_batch_one_object(df, transforms, batch_size)
        else:
            features, idxs_to_predict = self.extract_features_batch_two_objects(df, transforms, batch_size)
        batch_size = 65536
        predictions = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_data = features[i:i+batch_size]
                preds, _ = mlp_model(batch_data)
                # predictions.extend([bool(pred.item()) for pred in preds])
                predictions.extend(preds.cpu().tolist())

        all_predictions = [0] * len(df)
        for i, pred in zip(idxs_to_predict, predictions):
            all_predictions[i] = pred
        return all_predictions

    def extract_features_batch_one_object(self, df, transforms, batch_size):
        num_samples = len(df)
        all_features = []
        all_idxs_to_predict = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = batch_start + batch_size
            idxs_to_predict = []
            image_patches = []

            # Process each batch
            for i, (_, row) in enumerate(df.iloc[batch_start:batch_end].iterrows()):
                frame = row["img"]
                idxs_to_predict.append(batch_start + i)
                o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], (frame.shape[0], frame.shape[1]), factor=1)
                rois = [[0, o1_x1, o1_y1, o1_x2, o1_y2]]

                single_frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) # Shape: (1, C, H, W)

                rois_tensor = torch.tensor(rois, dtype=torch.float).to(self.device)
                # https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html
                # image_patches.shape: (K, C, 224, 224), where K is the number of bounding boxes
                patch_size=(224, 224)
                signle_frame_patches = ops.roi_align(single_frame, rois_tensor, output_size=patch_size, spatial_scale=1.0)
                image_patches.append(signle_frame_patches)

            image_patches = torch.cat(image_patches, dim=0)

            # Run CLIP model
            inputs = transforms(image_patches)
            with torch.no_grad():
                features = self.clip_model.get_image_features(pixel_values=inputs) # torch.FloatTensor of shape (batch_size, output_dim)

            all_features.append(features)
            all_idxs_to_predict.extend(idxs_to_predict)

        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)
        else:
            all_features = torch.tensor([], dtype=torch.float32)
        return all_features, all_idxs_to_predict

    def extract_features_batch_two_objects(self, df, transforms, batch_size):
        num_samples = len(df)
        all_features = []
        all_idxs_to_predict = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = batch_start + batch_size
            idxs_to_predict = []
            batch_boxes = []
            batch_o1_onames = []
            batch_o2_onames = []
            image_patches = []

            for i, (_, row) in enumerate(df.iloc[batch_start:batch_end].iterrows()):
                frame = row["img"]
                o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], (frame.shape[0], frame.shape[1]))
                o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], (frame.shape[0], frame.shape[1]))
                # Verify rois are correct
                roi_x1 = min(o1_x1, o2_x1)
                roi_y1 = min(o1_y1, o2_y1)
                roi_x2 = max(o1_x2, o2_x2)
                roi_y2 = max(o1_y2, o2_y2)
                if 0 <= roi_x1 < roi_x2 and 0 <= roi_y1 < roi_y2:
                    idxs_to_predict.append(batch_start + i)
                    rois = [[0, roi_x1, roi_y1, roi_x2, roi_y2]]
                    new_o1x1, new_o1y1, new_o1x2, new_o1y2, new_o2x1, new_o2y1, new_o2x2, new_o2y2 = self._compute_new_box_after_crop(row, (frame.shape[0], frame.shape[1]))
                    batch_boxes.append([int(new_o1x1), int(new_o1y1), int(new_o1x2), int(new_o1y2), int(new_o2x1), int(new_o2y1), int(new_o2x2), int(new_o2y2)])
                    batch_o1_onames.append(row['o1_oname'])
                    batch_o2_onames.append(row['o2_oname'])

                    single_frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) # Shape: (1, C, H, W)
                    rois_tensor = torch.tensor(rois, dtype=torch.float).to(self.device)
                    # https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html
                    # image_patches.shape: (K, C, 224, 224), where K is the number of bounding boxes
                    patch_size=(224, 224)
                    signle_frame_patches = ops.roi_align(single_frame, rois_tensor, output_size=patch_size, spatial_scale=1.0)
                    image_patches.append(signle_frame_patches)

            if len(image_patches) == 0:
                continue
            image_patches = torch.cat(image_patches, dim=0)
            # Run CLIP model
            batch_frames = image_patches.clone()
            # torch.tensor(image_patches).to(device)
            batch_boxes = torch.tensor(batch_boxes).to(self.device)
            N, C, H, W = batch_frames.shape
            # logger.debug(f"batch_frames.shape: {batch_frames.shape}, batch_boxes.shape: {batch_boxes.shape}")
            X = torch.arange(W, device=self.device).view(1, 1, W).expand(N, H, W)
            Y = torch.arange(H, device=self.device).view(1, H, 1).expand(N, H, W)
            subject_masks = (X >= batch_boxes[:, 0].view(N, 1, 1).expand(N, H, W)) & (X < batch_boxes[:, 2].view(N, 1, 1).expand(N, H, W)) & (Y >= batch_boxes[:, 1].view(N, 1, 1).expand(N, H, W)) & (Y < batch_boxes[:, 3].view(N, 1, 1).expand(N, H, W))
            target_masks = (X >= batch_boxes[:, 4].view(N, 1, 1).expand(N, H, W)) & (X < batch_boxes[:, 6].view(N, 1, 1).expand(N, H, W)) & (Y >= batch_boxes[:, 5].view(N, 1, 1).expand(N, H, W)) & (Y < batch_boxes[:, 7].view(N, 1, 1).expand(N, H, W))
            batch_frames_subject = batch_frames * subject_masks.unsqueeze(1).expand(N, C, H, W)
            batch_frames_target = batch_frames * target_masks.unsqueeze(1).expand(N, C, H, W)
            images = torch.cat([batch_frames, batch_frames_subject, batch_frames_target], dim=0) # (3N, C, H, W)
            inputs = transforms(images)
            if self.dataset == "charades":
                text_inputs = self.tokenizer(batch_o1_onames + batch_o2_onames, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(pixel_values=inputs) # torch.FloatTensor of shape (3N, 512)
                if self.dataset == "charades":
                    text_outputs = self.clip_model.get_text_features(**text_inputs) # torch.FloatTensor of shape (2N, 512)
            features = outputs.reshape(3, N, -1).permute(1, 0, 2).reshape(N, -1) # (N, 3 * 512)
            if self.dataset == "charades":
                text_features = text_outputs.reshape(2, N, -1).permute(1, 0, 2).reshape(N, -1) # (N, 2 * 512)
                features = torch.cat([features, text_features], dim=1) # (N, 5 * 512)
            all_features.append(features)
            all_idxs_to_predict.extend(idxs_to_predict)

        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)
        else:
            all_features = torch.tensor([], dtype=torch.float32)
        return all_features, all_idxs_to_predict

    # [[Deprecated]] Random access in Parquet is slow in general. Use predict_with_data instead.
    def predict_with_data_materialized(self, df, ckpt, n_obj):
        df = df.reset_index(drop=True)
        df['row_number'] = range(len(df))
        # Predict the labels of all the data points
        checkpoint = torch.load(ckpt)
        hyper_parameters = checkpoint["hyper_parameters"]
        mlp_model = mlp.MLPProd(**hyper_parameters)
        mlp_model.load_state_dict(checkpoint["state_dict"])
        mlp_model.eval()
        mlp_model.to(self.device)

        if n_obj == 1:
            df_with_features = self.conn.execute(f"""
                SELECT any_value(d.feature) as feature
                FROM df,
                    '{self.attribute_features_dir}/*.parquet' d
                WHERE df.vid=d.vid AND df.fid=d.fid
                    AND df.o1_oid=d.o1_oid
                GROUP BY df.row_number
                ORDER BY df.row_number
            """).df()
        else:
            df_with_features = self.conn.execute(f"""
                SELECT any_value(d.feature) as feature
                FROM df,
                    '{self.relationship_features_dir}/*.parquet' d
                WHERE df.vid=d.vid AND df.fid=d.fid
                    AND df.o1_oid=d.o1_oid AND df.o2_oid=d.o2_oid
                GROUP BY df.row_number
                ORDER BY df.row_number
            """).df()
        batch_size = 262144
        predictions = []
        with torch.no_grad():
            for i in range(0, len(df_with_features), batch_size):
                batch_data = df_with_features.iloc[i:i+batch_size]
                features = torch.tensor(batch_data["feature"].tolist(), dtype=torch.float32).to(self.device)
                preds = mlp_model(features)
                # predictions.extend([bool(pred.item()) for pred in preds])
                predictions.extend(preds.cpu().tolist())
        # predictions = []
        # with torch.no_grad():
        #     for _, row in tqdm(df_with_features.iterrows(), total=len(df_with_features), file=sys.stdout, desc="MLP Predicting"):
        #         feature = torch.tensor(row["feature"], dtype=torch.float32).to(self.device)
        #         pred = mlp_model(feature)
        #         predictions.append(bool(pred.item()))
        return predictions

    def compute_udf_score(
        self,
        gt_udf_name,
        udf_candidate,
        udf_name,
        n_obj,
        df,
        df_newly_labeled=None,
        add_one=False,
    ):
        """
        Compute the F1 score of the UDF candidate on the data (train or test), using the ground truth UDF as the label
        if df_newly_labeled is provided, also compute the number of misclassified samples of them (which is used to compute loss_t)
        """
        if udf_candidate.id == "model":
                best_ckpt = udf_candidate.function_implementation
                # logger.debug("df before predict_with_data: {}".format(df[["vid", "fid", "o1_oid"]].to_string()))
                y_pred = self.predict_with_data(df, best_ckpt, n_obj)
                # logger.debug(f"y_pred: {y_pred}")
                # logger.debug("df after predict_with_data: {}".format(df[["vid", "fid", "o1_oid"]].to_string()))
                if df_newly_labeled is not None:
                    y_pred_new = self.predict_with_data(df_newly_labeled, best_ckpt, n_obj)
                    # logger.debug(f"y_pred_new: {y_pred_new}")
        else:
            try:
                # For each sampled row in df, construct o1 and o2
                kwargs = {}
                for k, v in udf_candidate.kwargs.items():
                    kwargs[k] = float(v)
                py_func_name = "py_{}".format(udf_name)
                exec(udf_candidate.function_implementation, globals())
                udf_obj = globals()[py_func_name]
                y_pred = self.exec_udf_with_data(df, udf_obj, kwargs, n_obj, requires_no_error=False, timeout=max(60, len(df)*0.2))
                if df_newly_labeled is not None:
                    y_pred_new = self.exec_udf_with_data(df_newly_labeled, udf_obj, kwargs, n_obj, requires_no_error=False, timeout=max(60, len(df_newly_labeled)*0.2))
            except Exception as e:
                logger.exception("ERROR: failed to execute udf_candidate {}: {}".format(udf_candidate.id, e))
                raise
                # y_pred = [False] * len(df)
                # if df_newly_labeled is not None:
                #     y_pred_new = [False] * len(df_newly_labeled)

        # Compute y_true and f1 score
        if n_obj == 1:
            # y_true = df.apply(lambda row: gt_udf_name in row["o1_gt_anames"], axis=1)
            y_true = [gt_udf_name in o1_gt_anames for o1_gt_anames in df['o1_gt_anames']]
        elif n_obj == 2:
            # y_true = df.apply(lambda row: gt_udf_name in row["o1_o2_gt_rnames"], axis=1)
            y_true = [gt_udf_name in o1_o2_gt_rnames for o1_o2_gt_rnames in df['o1_o2_gt_rnames']]
        # logger.debug(f"y_true: {y_true}, y_pred: {y_pred}")
        if add_one:
            # Add one TP prediction to the model
            y_true.append(True)
            y_pred.append(True)
        score = f1_score(y_true, y_pred, zero_division=0.0)
        logger.info("udf_candidate: {}, score: {}".format(udf_candidate.id, score))
        # logger.info("y_true: {}, y_pred: {}".format(y_true, y_pred))
        logger.info("predicted positive: {}, predicted negative: {}".format(sum(y_pred), len(y_pred) - sum(y_pred)))
        logger.info("positive: {}, negative: {}".format(sum(y_true), len(y_true) - sum(y_true)))

        # Compute y_true_new and num_misclassified
        if df_newly_labeled is not None:
            if n_obj == 1:
                # y_true_new = df_newly_labeled.apply(
                #     lambda row: gt_udf(row["o1_oname"], row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"], row["o1_anames"], row["height"], row["width"]), axis=1
                # )
                y_true_new = [gt_udf_name in o1_gt_anames for o1_gt_anames in df_newly_labeled['o1_gt_anames']]
            elif n_obj == 2:
                # y_true_new = df_newly_labeled.apply(
                #     lambda row: gt_udf(row["o1_oname"], row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"], row["o1_anames"], row["o2_oname"], row["o2_x1"], row["o2_y1"], row["o2_x2"], row["o2_y2"], row["o2_anames"], row["o1_o2_rnames"], row["o2_o1_rnames"], row["height"], row["width"]), axis=1
                # )
                y_true_new = [gt_udf_name in o1_o2_gt_rnames for o1_o2_gt_rnames in df_newly_labeled['o1_o2_gt_rnames']]
            # Count the number of misclassifications for the new samples
            # logger.debug(f"y_true_new: {y_true_new}, y_pred_new: {y_pred_new}")
            # logger.debug(f"y_true_new: {y_true_new}, y_pred_new: {y_pred_new}")
            num_misclassified = np.sum(np.array(y_true_new != y_pred_new) * 1)
            return score, num_misclassified
        else:
            return score
