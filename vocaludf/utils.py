import duckdb
import itertools
import copy
from vocaludf.parser import parse_udf
import json
import logging
import os
from torch.utils.data import IterableDataset
import torch
import pandas as pd
import math
from typing import List
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from PIL import Image
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, LlavaNextForConditionalGeneration, LlavaNextProcessor

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_COST = {
    "gpt-4o": [2.5 / 1e6, 10 / 1e6], # input cost per token, output cost per token
    "gpt-4o-2024-08-06": [2.5 / 1e6, 10 / 1e6],
    "gpt-4-turbo-2024-04-09": [10 / 1e6, 30 / 1e6],
}

RESOLVE_MODEL_NAME = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-2024-08-06": "gpt-4o-2024-08-06",
    "gpt-4-turbo-2024-04-09": "gpt-4-turbo-2024-04-09",
}

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def exception_hook(exc_type, exc_value, exc_traceback, logger=logger):
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )

def setup_logging(config, base_dir, log_filename, logger):
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        base_dir
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), mode="w")
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
        query_filename,
        query_id,
        run_id,
        num_workers,
        save_labeled_data,
        load_labeled_data,
        n_train_distill,
        selection_strategy,
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
        self.query_filename = query_filename
        self.query_id = query_id
        self.run_id = run_id
        self.num_workers = num_workers
        self.selection_strategy = selection_strategy
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
        self.n_train_selection = self.config[self.dataset]["n_train_selection"]
        self.n_test_selection = self.config[self.dataset]["n_test_selection"]

        # Initialization for model distillation
        self.n_train_distill = n_train_distill
        self.n_test_distill = self.config[self.dataset]["n_test_distill"]
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
        metadata_join_clause = '' if self.dataset in ['clevrer'] else f'LEFT OUTER JOIN {self.dataset}_metadata m ON o1.vid = m.vid AND o1.fid = m.fid'
        height_width_clause = '320 AS height, 480 AS width' if self.dataset in ['clevrer'] else 'm.height AS height, m.width AS width'
        group_by_clause = 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2' if self.dataset in ['clevrer'] else 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2, m.height, m.width'
        obj_parameters = ','.join('?' for _ in self.object_domain)
        attr_parameters = ','.join('?' for _ in self.attribute_domain)
        dataset_size = self.config[self.dataset]["dataset_size"]
        # Use the first half of the dataset for training
        where_clause = f"WHERE o1.oname = ANY([{obj_parameters}]) AND o1.vid < {dataset_size // 2}"
        sql = f"""
            SELECT
                o1.vid AS vid, o1.fid AS fid, o1.oid AS o1_oid, o1.oname AS o1_oname,
                o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                COALESCE(ARRAY_AGG(DISTINCT a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
                COALESCE(ARRAY_AGG(DISTINCT ap.aname) FILTER (WHERE ap.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
                {height_width_clause}
            FROM {self.dataset}_objects o1
            LEFT OUTER JOIN {self.dataset}_attributes a ON o1.vid = a.vid AND o1.fid = a.fid AND o1.oid = a.oid
            LEFT OUTER JOIN {self.dataset}_attribute_predictions ap ON o1.vid = ap.vid AND o1.fid = ap.fid AND o1.oid = ap.oid
            {metadata_join_clause}
            {where_clause}
            GROUP BY {group_by_clause}
        """
        # ORDER BY o1.vid, o1.fid, o1.oid, o1.x1, o1.y1, o1.x2, o1.y2
        logger.debug(f"Create one_object table:\n{sql}")
        one_object_df = self.conn.execute(sql, self.attribute_domain + self.object_domain).df()

        rel_parameters = ','.join('?' for _ in self.relationship_domain)
        # Use the first half of the dataset for training
        obj_attr_where_clause = f"WHERE o.oname = ANY([{obj_parameters}]) AND o.vid < {dataset_size // 2}"
        rel_where_clause = f"WHERE vid < {dataset_size // 2}"
        sql = f"""
            WITH obj_with_attrs AS (
                SELECT
                    o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                    COALESCE(ARRAY_AGG(DISTINCT a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS attributes
                FROM {self.dataset}_objects o
                LEFT OUTER JOIN {self.dataset}_attribute_predictions a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid AND a.aname = ANY([{attr_parameters}])
                {obj_attr_where_clause}
                GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
            )
            , relationships_expanded AS (
                SELECT
                    vid, fid, oid1, oid2,
                    ARRAY_AGG(DISTINCT rname) AS gt_rnames
                FROM {self.dataset}_relationships
                {rel_where_clause}
                GROUP BY vid, fid, oid1, oid2
            )
            , relationship_predictions_expanded AS (
                SELECT
                    vid, fid, oid1, oid2,
                    COALESCE(ARRAY_AGG(DISTINCT rname) FILTER (WHERE rname = ANY([{rel_parameters}])), ARRAY[]::varchar[]) AS rnames
                FROM {self.dataset}_relationship_predictions
                {rel_where_clause}
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
        """
        # ORDER BY o1.vid, o1.fid, o1.oid, o2.oid, o1.x1, o1.y1, o1.x2, o1.y2, o2.x1, o2.y1, o2.x2, o2.y2
        logger.debug(f"Create two_objects table:\n{sql}")
        two_objects_df = self.conn.execute(sql, self.attribute_domain + self.object_domain + self.relationship_domain).df()
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

# TODO: We can also support video loading and feature extraction on the fly, without creating the parquet files,
# since writing parquet files is the bottleneck right now.
class PredImageDataset(IterableDataset):
    def __init__(self, conn, n_obj, attribute_features_dir, relationship_features_dir):
        self.conn = conn
        self.n_obj = n_obj
        self.features_dir = attribute_features_dir if n_obj == 1 else relationship_features_dir

        self.files = self._get_files(self.features_dir)

        self.num_files = len(self.files)
        self.start = 0
        self.end = self.num_files
        self.len = duckdb.sql(f"SELECT COUNT(*) FROM '{self.features_dir}/*.parquet';").fetchall()[0][0]
        self.columns = ['vid', 'fid', 'o1_oid', 'feature'] if n_obj == 1 else ['vid', 'fid', 'o1_oid', 'o2_oid', 'feature']
    def _get_files(self, features_dir, extension=".parquet") -> List[str]:
        all_files = os.listdir(features_dir)
        matched_files = sorted([os.path.join(features_dir, f) for f in all_files if f.endswith(extension)])
        return matched_files

    def __len__(self):
        return self.len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else: # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        for file in self.files[iter_start:iter_end]:
            df = pd.read_parquet(file, columns=self.columns)

            feature_col = df["feature"].values
            metadata = df.drop('feature', axis=1)

            for (_, row), feature in zip(metadata.iterrows(), feature_col):
                yield np.array(row.to_numpy(dtype=int)), torch.tensor(feature, dtype=torch.float32)


def expand_box(x1,y1,x2,y2,img_size,factor=1.5):
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

def parse_signature(signature):
    """
    Example:
    signature: "Color_red(o1, -1)"
    parsed result: {'fn_name': 'Color_red', 'variables': ['o1'], 'parameter': -1}
    """
    # NOTE: could throw an exception if the signature is not in the correct format
    result = parse_udf().parseString(signature, parseAll=True).as_dict()
    udf_name = result["fn_name"]
    udf_vars = result["variables"]
    # tokens = list(tokenize.generate_tokens(io.StringIO(signature).readline))
    # udf_name = tokens[0].string
    # udf_vars = [token for token in tokens[2:-3] if token.string not in [',','=']]
    return udf_name, udf_vars

def transform_function(original_code, instantiation_dict):
    """
    Transforms the original code by removing **kwargs from the function definition
    and inserting a line defining kwargs with the corrected string format.

    Args:
        original_code (str): The original code to be transformed.
        instantiation_dict (dict): The dictionary containing the values for kwargs.

    Returns:
        str: The transformed code.
    """
    # Split the original function into lines
    lines = original_code.split('\n')

    # Find the line with the function definition and remove **kwargs
    for i, line in enumerate(lines):
        if line.startswith('def ') and '**kwargs' in line:
            # Replace **kwargs with nothing
            lines[i] = line.replace(', **kwargs', '').replace('**kwargs, ', '').replace('**kwargs', '')

            # Insert the line defining kwargs with the corrected string format
            kwargs_str = json.dumps(instantiation_dict)
            kwargs_line = f"    kwargs = {kwargs_str}"
            lines.insert(i + 1, kwargs_line)
            break

    # Rejoin the modified lines into a single string
    transformed_code = '\n'.join(lines)

    return transformed_code

def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key +"}}", value.replace('"', "'"))
    return text

def get_active_domain(config, dataset, registered_functions):
    object_domain = config[dataset]['onames']
    relationship_domain = []
    attribute_domain = []

    for registered_function in registered_functions:
        signature = registered_function["signature"]
        registered_function_name, registered_function_vars = parse_signature(signature)
        if registered_function_name.lower() == "object":
            continue
        if len(registered_function_vars) == 2:
            # Relationship UDF
            relationship_domain.append(registered_function_name.lower())
        else:
            # Attribute UDF
            attribute_domain.append(registered_function_name.lower())
    return object_domain, relationship_domain, attribute_domain

def duckdb_execute_video_materialize(conn, current_query, input_vids, available_udf_names, materialized_udf_names, on_the_fly_udf_names):
    """
    There are three types of UDFs:
    - available_udf_names: available UDFs
    - materialized_udf_names: new UDFs, with results already materialized
    - on_the_fly_udf_names: new UDFs, with results computed on-the-fly
    """
    # Duration((LeftOf(o2, o0), Color_Red(o0), FrontOf(o1, o2)), 15); Duration((FarFrom(o1, o2), LeftOf(o0, o2)), 5); (Behind(o2, o0), Material_Metal(o1))

    output_vids = []

    temp_tables = []
    # select input videos
    if input_vids is None:
        conn.execute(f"CREATE TEMPORARY TABLE obj_attr_filtered AS SELECT * FROM one_object;")
        conn.execute(f"CREATE TEMPORARY TABLE rel_filtered AS SELECT * FROM two_objects;")
    elif isinstance(input_vids, int):
        conn.execute(f"CREATE TEMPORARY TABLE obj_attr_filtered AS SELECT * FROM one_object WHERE vid < {input_vids};")
        conn.execute(f"CREATE TEMPORARY TABLE rel_filtered AS SELECT * FROM two_objects WHERE vid < {input_vids};")
    else: # list of input_vids
        parameters = ','.join('?' for _ in input_vids)
        conn.execute(f"CREATE TEMPORARY TABLE obj_attr_filtered AS SELECT * FROM one_object WHERE vid = ANY([{parameters}]);", input_vids)
        conn.execute(f"CREATE TEMPORARY TABLE rel_filtered AS SELECT * FROM two_objects WHERE vid = ANY([{parameters}]);", input_vids)
    temp_tables.append("obj_attr_filtered")
    temp_tables.append("rel_filtered")

    encountered_variables_prev_graphs = []
    encountered_variables_current_graph = []
    for graph_idx, dict in enumerate(current_query):
        # Generate scene graph:
        scene_graph = dict["scene_graph"]
        duration_constraint = dict["duration_constraint"]
        for p in scene_graph:
            for v in p["variables"]:
                if v not in encountered_variables_current_graph:
                    encountered_variables_current_graph.append(v)

        # Execute for unseen videos
        encountered_variables_current_graph = sorted(encountered_variables_current_graph, key=lambda x: int(x[1:]))
        table_list = ["obj_attr_filtered as {}".format(v) for v in encountered_variables_current_graph]
        where_clauses = []
        for i in range(len(encountered_variables_current_graph)-1):
            # [where condition] All obj_attr_filtered tables should have same vid and fid
            where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_current_graph[i], v2=encountered_variables_current_graph[i+1])) # join variables
        for p in scene_graph:
            predicate = p["predicate"]
            parameter = p.get("parameter", None) # only object predicates have parameters, which are class names
            variables = p["variables"]
            if predicate != "object":
                assert parameter is None
            if predicate == "object":
                # [where condition] object
                where_clauses.append(f"{variables[0]}.o1_oname = '{parameter}'")
            elif predicate in available_udf_names:
                if len(variables) == 1:
                    # [where condition] available attribute UDFs
                    where_clauses.append(f"'{predicate}' = ANY({variables[0]}.o1_anames)")
                else:
                    v0 = variables[0]
                    v1 = variables[1]
                    # [where condition] available relationship UDFs
                    pred_table = f"{v0}_{predicate}_{v1}"
                    table_list.append(f"rel_filtered as {pred_table}")
                    where_clauses.append(f"""
                        {v0}.vid = {pred_table}.vid
                        and {v0}.fid = {pred_table}.fid
                        and {v0}.o1_oid = {pred_table}.o1_oid
                        and {v1}.o1_oid = {pred_table}.o2_oid
                        and '{predicate}' = ANY({pred_table}.o1_o2_rnames)
                    """)
            elif f"udf_{predicate}" in materialized_udf_names:
                if len(variables) == 1:
                    # [where condition] new attribute UDFs (materialized)
                    v0 = variables[0]
                    pred_table = f"{v0}_{predicate}"
                    table_list.append(f"udf_{predicate} as {pred_table}")
                    where_clauses.append(f"""
                        {v0}.vid = {pred_table}.vid
                        and {v0}.fid = {pred_table}.fid
                        and {v0}.o1_oid = {pred_table}.o1_oid
                        and {pred_table}.pred = 1
                    """)
                else:
                    # [where condition] new relationship UDFs (materialized)
                    v0 = variables[0]
                    v1 = variables[1]
                    pred_table = f"{v0}_{predicate}_{v1}"
                    table_list.append(f"udf_{predicate} as {pred_table}")
                    where_clauses.append(f"""
                        {v0}.vid = {pred_table}.vid
                        and {v0}.fid = {pred_table}.fid
                        and {v0}.o1_oid = {pred_table}.o1_oid
                        and {v1}.o1_oid = {pred_table}.o2_oid
                        and {pred_table}.pred = 1
                    """)
            elif predicate in on_the_fly_udf_names:
                if len(variables) == 1:
                    # [where condition] new attribute UDFs (on-the-fly)
                    v0 = variables[0]
                    where_clauses.append(f"""
                        {predicate}({v0}.o1_oname, {v0}.o1_x1, {v0}.o1_y1, {v0}.o1_x2, {v0}.o1_y2, {v0}.o1_anames, {v0}.height, {v0}.width) = true
                    """)
                else:
                    # [where condition] new relationship UDFs (on-the-fly)
                    v0 = variables[0]
                    v1 = variables[1]
                    pred_table = f"{v0}_{predicate}_{v1}"
                    table_list.append(f"rel_filtered as {pred_table}")
                    where_clauses.append(f"""
                        {v0}.vid = {pred_table}.vid
                        and {v0}.fid = {pred_table}.fid
                        and {v0}.o1_oid = {pred_table}.o1_oid
                        and {v1}.o1_oid = {pred_table}.o2_oid
                        and {predicate}({pred_table}.o1_oname, {pred_table}.o1_x1, {pred_table}.o1_y1, {pred_table}.o1_x2, {pred_table}.o1_y2, {pred_table}.o1_anames, {pred_table}.o2_oname, {pred_table}.o2_x1, {pred_table}.o2_y1, {pred_table}.o2_x2, {pred_table}.o2_y2, {pred_table}.o2_anames, {pred_table}.o1_o2_rnames, {pred_table}.o2_o1_rnames, {pred_table}.height, {pred_table}.width) = true
                    """)
            else:
                raise ValueError("Unknown predicate: {}".format(predicate))
                # TODO: for robustness, we should remove it from the query and continue execution
        # [where condition] Different obj_attr_filtered tables should have different oids
        for var_pair in itertools.combinations(encountered_variables_current_graph, 2):
            where_clauses.append("{}.o1_oid <> {}.o1_oid".format(var_pair[0], var_pair[1]))

        if graph_idx > 0:
            fields = "{v}.vid as vid, {v}.fid as fid".format(v=encountered_variables_current_graph[0])
            # fields += ", ".join(["{v}.o1_oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
            obj_union = copy.deepcopy(encountered_variables_prev_graphs)
            obj_union_fields = []
            obj_intersection_fields = []
            for v in encountered_variables_prev_graphs:
                obj_union_fields.append(f"t0.{v}_oid as {v}_oid")
            for v in encountered_variables_current_graph:
                if v in encountered_variables_prev_graphs:
                    obj_intersection_fields.append(f"t0.{v}_oid = {v}.o1_oid")
                else:
                    for u in encountered_variables_prev_graphs:
                        obj_intersection_fields.append(f"t0.{u}_oid <> {v}.o1_oid")
                    obj_union.append(v)
                    obj_union_fields.append(f"{v}.o1_oid as {v}_oid")
            obj_union_fields = ", ".join(obj_union_fields)
            obj_intersection_fields = " and ".join(obj_intersection_fields)
            where_clauses.append(f"""
                t0.vid = {encountered_variables_current_graph[0]}.vid
                AND {obj_intersection_fields}
                AND t0.fid < {encountered_variables_current_graph[0]}.fid
            """)
            where_clauses = " and ".join(where_clauses)
            table_list.append(f"g{graph_idx-1}_contiguous t0")
            table_str = ", ".join(table_list)
            sql_string = f"""
                CREATE TEMPORARY TABLE g{graph_idx} AS
                SELECT DISTINCT {fields}, {obj_union_fields}
                FROM {table_str}
                WHERE {where_clauses};
            """
            logger.debug(sql_string)
            conn.execute(sql_string)
            temp_tables.append(f"g{graph_idx}")
        else:
            fields = "{v}.vid as vid, {v}.fid as fid, ".format(v=encountered_variables_current_graph[0])
            fields += ", ".join(["{v}.o1_oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
            where_clauses = " and ".join(where_clauses)
            table_str = ", ".join(table_list)
            sql_string = f"""
                CREATE TEMPORARY TABLE g{graph_idx} AS
                SELECT DISTINCT {fields}
                FROM {table_str}
                WHERE {where_clauses};
            """
            logger.debug(sql_string)
            conn.execute(sql_string)
            temp_tables.append(f"g{graph_idx}")
            obj_union = encountered_variables_current_graph

        # Generate scene graph sequence:
        table_name = f"g{graph_idx}"
        obj_union_fields = ", ".join(["{}_oid".format(v) for v in obj_union])
        sql_string = """
            CREATE TEMPORARY TABLE g{graph_idx}_windowed AS (
            SELECT DISTINCT vid, fid, {obj_union_fields},
            lead(fid, {duration_constraint} - 1, 0) OVER (PARTITION BY vid, {obj_union_fields} ORDER BY fid) as fid_offset
            FROM {table_name}
        );
        """.format(graph_idx=graph_idx, duration_constraint=duration_constraint, obj_union_fields=obj_union_fields, table_name=table_name)
        logger.debug(sql_string)
        conn.execute(sql_string)
        temp_tables.append(f"g{graph_idx}_windowed")

        sql_string = """
            CREATE TEMPORARY TABLE g{graph_idx}_contiguous AS (
            SELECT DISTINCT vid, {obj_union_fields}, min(fid_offset) AS fid
            FROM g{graph_idx}_windowed
            WHERE fid_offset = fid + ({duration_constraint} - 1)
            GROUP BY vid, {obj_union_fields}
        );
        """.format(graph_idx=graph_idx, obj_union_fields=obj_union_fields, duration_constraint=duration_constraint)
        logger.debug(sql_string)
        conn.execute(sql_string)
        temp_tables.append(f"g{graph_idx}_contiguous")
        conn.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(graph_idx))
        res = conn.fetchall()
        output_vids = [row[0] for row in res]
        encountered_variables_prev_graphs = obj_union
        encountered_variables_current_graph = []
    # Drop tables
    for temp_table in temp_tables:
        conn.execute(f"DROP TABLE {temp_table}")
    return output_vids
