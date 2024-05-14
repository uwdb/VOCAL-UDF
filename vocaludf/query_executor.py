import random
import string
import os
from vocaludf.utils import (
    duckdb_execute_cache_sequence,
    duckdb_execute_clevrer_cache_sequence,
    duckdb_execute_clevrer_materialize,
    parse_signature,
    PredImageDataset,
)
from vocaludf.pretrained_model_api import image_captioning, visual_question_answering, depth_estimation
import time
import duckdb
import logging
from sklearn.metrics import f1_score
from typing import List
import torch
from vocaludf import mlp
from torch.utils.data import IterableDataset
import math
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import repeat
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def VideoFrameDaliDataloader(
    vids,
    sequence_length=64,
    video_directory="/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/",
    device='gpu',
    batch_size=None,
    num_threads=None,
):
    assert device == 'gpu', 'dali video_resize only supports gpu backend'

    video_files = [
        os.path.join(
            video_directory,
            f"video_{str(vid//1000*1000).zfill(5)}-{str((vid//1000+1)*1000).zfill(5)}",
            f"video_{str(vid).zfill(5)}.mp4",
        )
        for vid in vids
    ]

    @pipeline_def
    def video_pipe(filenames, vids):
        videos, labels, start_frame_num = fn.readers.video(
            device="gpu",
            filenames=filenames,
            # the only "boosting parameter" is the sequence_length: https://github.com/NVIDIA/DALI/issues/4498
            sequence_length=sequence_length,
            pad_sequences=True,
            # shard_id=0,
            # num_shards=1,
            dtype=types.UINT8,
            random_shuffle=False,
            initial_fill=None, # Only relevant when shuffle=True
            file_list_include_preceding_frame=False, # Quiet warning about default changing
            dont_use_mmap=True,
            skip_vfr_check=True,
            enable_frame_num=True,
            labels=vids,
            name='reader',
        )
        return videos, labels, start_frame_num

    pipe = video_pipe(batch_size=batch_size, num_threads=num_threads, device_id=0, filenames=video_files, vids=vids)
    pipe.build()
    return pipe


class QueryExecutor:
    def __init__(self, config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_udf_names, on_the_fly_udf_names, program_with_pixels, num_workers):
        self.config = config
        self.dataset = dataset
        self.object_domain = object_domain
        self.attribute_domain = attribute_domain
        self.relationship_domain = relationship_domain
        self.conn = duckdb.connect(
            database=os.path.join(config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )
        self.registered_functions = registered_functions
        self.available_udf_names = available_udf_names
        self.materialized_udf_names = materialized_udf_names
        self.on_the_fly_udf_names = on_the_fly_udf_names
        logger.info(f"available_udf_names: {self.available_udf_names}")
        logger.info(f"materialized_udf_names: {self.materialized_udf_names}")
        logger.info(f"on_the_fly_udf_names: {self.on_the_fly_udf_names}")
        self.program_with_pixels = program_with_pixels
        self.attribute_features_dir = os.path.join(self.config[self.dataset]["features_dir"], "attribute")
        self.relationship_features_dir = os.path.join(self.config[self.dataset]["features_dir"], "relationship")
        self.num_workers = num_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.init_table()
        self.materialized_udfs = {}
        for func in self.registered_functions:
            signature = func["signature"]
            udf_name, udf_vars = parse_signature(signature)
            if func.get("semantic_interpretation", "") == "dummy":
                if len(udf_vars) == 1:
                    df = self.conn.execute(f"SELECT vid, fid, o1_oid, 1 as pred FROM one_object").df()
                else:
                    df = self.conn.execute(f"SELECT vid, fid, o1_oid, o2_oid, 1 as pred FROM two_objects").df()
                self.materialized_udfs[udf_name] = df
            elif func.get("semantic_interpretation", "") == "model":
                df = self.get_materialized_df(func)
                self.materialized_udfs[udf_name] = df
            else:
                lines = func["function_implementation"].split("\n")
                for i, line in enumerate(lines):
                    if line.startswith('def '):
                        python_func_name = line.split()[1].split("(")[0]
                        # create a unique suffix
                        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
                        python_func_args = line.split("(")[1].split(")")[0].split(",")
                        # Add type annotations to the function header
                        # for attribute: attr(o1_oname: str, o1_x1: int, o1_y1: int, o1_x2: int, o1_y2: int, o1_anames: List[str], height: int, width: int)
                        # for relationship: rel(o1_oname: str, o1_x1: int, o1_y1: int, o1_x2: int, o1_y2: int, o1_anames: List[str], o2_oname: str, o2_x1: int, o2_y1: int, o2_x2: int, o2_y2: int, o2_anames: List[str], o1_o2_rnames: List[str], o2_o1_rnames: List[str], height: int, width: int)
                        if len(python_func_args) == 8:
                            # attribute
                            types = ["str", "int", "int", "int", "int", "List[str]", "int", "int"]
                        elif len(python_func_args) == 9:
                            # attribute + img
                            types = ["np.ndarray", "str", "int", "int", "int", "int", "List[str]", "int", "int"]
                        elif len(python_func_args) == 16:
                            # relationship
                            types = ["str", "int", "int", "int", "int", "List[str]", "str", "int", "int", "int", "int", "List[str]", "List[str]", "List[str]", "int", "int"]
                        elif len(python_func_args) == 17:
                            # relationship + img
                            types = ["np.ndarray", "str", "int", "int", "int", "int", "List[str]", "str", "int", "int", "int", "int", "List[str]", "List[str]", "List[str]", "int", "int"]
                        else:
                            raise ValueError("Unknown number of arguments in the function header: {}".format(len(python_func_args)))
                        python_arg_str = ", ".join([f"{arg}: {type}" for arg, type in zip(python_func_args, types)])
                        python_header_type_annotated = f"def {python_func_name}_{suffix}({python_arg_str}) -> bool:"
                        # python_header_type_annotated = f"def {python_func_name}_{suffix}({python_arg_str}) -> bool:"
                        lines[i] = python_header_type_annotated
                        break
                # Rejoin the modified lines into a single string
                function_implementation = '\n'.join(lines)
                exec(function_implementation)
                exec(
                    f"self.conn.create_function('{udf_name}', {python_func_name}_{suffix})"
                )
            logger.debug(f"Registered function: {signature}")


    def init_table(self):
        attr_parameters = ','.join('?' for _ in self.attribute_domain)
        sql = f"""
            CREATE TEMPORARY TABLE one_object AS
            SELECT
                o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
                o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
                320 AS height, 480 AS width
            FROM {self.dataset}_objects o
            LEFT OUTER JOIN {self.dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
            GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
        """
        logger.debug(f"Create one_object table:\n{sql}")
        self.conn.execute(sql, self.attribute_domain).df()

        rel_parameters = ','.join('?' for _ in self.relationship_domain)
        sql = f"""
            CREATE TEMPORARY TABLE two_objects AS
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
                320 AS height, 480 AS width
            FROM obj_with_attrs o1
            JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid
            LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
            LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
            WHERE o1.oid <> o2.oid
        """
        logger.debug(f"Create two_objects table:\n{sql}")
        self.conn.execute(sql, self.attribute_domain + self.relationship_domain).df()

    def get_materialized_df(self, func):
        best_ckpt = func["function_implementation"]
        # Predict the labels of all the data points
        checkpoint = torch.load(best_ckpt)
        hyper_parameters = checkpoint["hyper_parameters"]
        best_mlp_model = mlp.MLPProd(**hyper_parameters)
        best_mlp_model.load_state_dict(checkpoint["state_dict"])
        best_mlp_model.eval()
        best_mlp_model.to(self.device)

        signature = func["signature"]
        udf_name, udf_vars = parse_signature(signature)
        n_obj = len(udf_vars)

        pred_dataset = PredImageDataset(self.conn, n_obj, self.attribute_features_dir, self.relationship_features_dir)
        pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=262144, num_workers=self.num_workers, shuffle=False)

        rows = []
        predictions = []
        with torch.no_grad():
            for row, feature in tqdm(pred_loader, file=sys.stdout):
                feature = feature.to(self.device)
                pred = best_mlp_model(feature)
                rows.extend(row.tolist())
                predictions.extend(pred.cpu().tolist())
        columns = ['vid', 'fid', 'o1_oid'] if n_obj == 1 else ['vid', 'fid', 'o1_oid', 'o2_oid']
        df_with_pred = pd.DataFrame(rows, columns=columns)
        df_with_pred['pred'] = predictions

        return df_with_pred

    def run(self, program, y_true, debug=False):
        # Bind each dataframe to a variable name, so that we can refer to them in the SQL query
        for udf_name, df in self.materialized_udfs.items():
            exec(f"{udf_name} = df")
        if self.dataset == "clevrer":
            exec_func = duckdb_execute_clevrer_materialize
            if debug:
                input_vids = 1000
            else:
                input_vids = 10000
        elif self.dataset == "clevr":
            exec_func = duckdb_execute_cache_sequence
            if debug:
                input_vids = 1500
            else:
                input_vids = 15000
        else:
            raise ValueError(
                "Unknown dataset: {}".format(self.dataset)
            )
        logger.info("Running query: {}".format(program["query"]))
        memo = [{} for _ in range(72159)]  # Not used

        if self.program_with_pixels:
            _start = time.time()
            # First, execute the query with all on-the-fly UDFs removed
            query_program = self.remove_on_the_fly_udfs(program["query"])
            result, new_memo = exec_func(
                self.conn,
                query_program,
                memo,
                input_vids,
                available_udf_names=self.available_udf_names,
                materialized_udf_names=self.materialized_udf_names,
                on_the_fly_udf_names=[]
            )
            logger.info("Time to execute first query: {}".format(time.time() - _start))
            result = sorted(result)
            logger.info("output vids: {}".format(result))
            self.materialize_on_the_fly_udfs(result)
            for udf_name, df in self.materialized_udfs.items():
                exec(f"{udf_name} = df")
            _start = time.time()
            # Then, execute the query with all on-the-fly UDFs over the result of the previous query
            result, new_memo = exec_func(
                self.conn,
                program["query"],
                memo,
                result, # matching vids from the previous query
                available_udf_names=self.available_udf_names,
                materialized_udf_names=self.materialized_udf_names,
                on_the_fly_udf_names=self.on_the_fly_udf_names
            )
            logger.info("Time to execute second query: {}".format(time.time() - _start))
            result = sorted(result)
            logger.info("output vids: {}".format(result))
        else:
            _start = time.time()
            result, new_memo = exec_func(
                self.conn,
                program["query"],
                memo,
                input_vids,
                available_udf_names=self.available_udf_names,
                materialized_udf_names=self.materialized_udf_names,
                on_the_fly_udf_names=self.on_the_fly_udf_names
            )
            logger.info("Time to execute query: {}".format(time.time() - _start))
            result = sorted(result)
            logger.info("output vids: {}".format(result))
        # logger.info("output vids: {}".format(result))
        y_pred = [1 if i in result else 0 for i in range(input_vids)]

        # logger.info("predictions: {}".format(y_pred))
        # logger.info("true labels: {}".format(y_true[:input_vids]))
        f1 = f1_score(y_true[:input_vids], y_pred)
        logger.info("F1 score: {}".format(f1))
        return y_pred

    def remove_on_the_fly_udfs(self, query):
        target_query = []
        for q in query:
            target_scene_graph = []
            for sg in q["scene_graph"]:
                if sg["predicate"] not in self.on_the_fly_udf_names:
                    target_scene_graph.append(sg)
            if len(target_scene_graph) > 0:
                target_query.append({"scene_graph": target_scene_graph, "duration_constraint": q["duration_constraint"]})
        return target_query

    def materialize_on_the_fly_udfs(self, vids):
        logger.info("Start materializing on-the-fly UDFs")

        # Filter on-the-fly UDFs
        on_the_fly_udfs = [func for func in self.registered_functions if parse_signature(func["signature"])[0] in self.on_the_fly_udf_names]

        logger.info("filtering tables by matching vids")
        # Group one_object and two_objects tables by vid and fid
        parameters = ','.join('?' for _ in vids)
        df_one_object = self.conn.execute(f"SELECT * FROM one_object WHERE vid = ANY([{parameters}])", vids).df()
        df_one_object_grouped = df_one_object.groupby(['vid', 'fid'], as_index=False, sort=False)
        df_two_objects = self.conn.execute(f"SELECT * FROM two_objects WHERE vid = ANY([{parameters}])", vids).df()
        df_two_objects_grouped = df_two_objects.groupby(['vid', 'fid'], as_index=False, sort=False)
        logger.info(f"df_one_object shape: {df_one_object.shape}")
        logger.info(f"df_two_objects shape: {df_two_objects.shape}")
        udf_map = {}
        for func in on_the_fly_udfs:
            udf_name, udf_vars = parse_signature(func["signature"])
            n_obj = len(udf_vars)
            py_func_name = "py_{}".format(udf_name)
            exec(func["function_implementation"], globals())
            udf_obj = globals()[py_func_name]
            udf_map[udf_name] = (udf_obj, n_obj)

        logger.info("building video dataloader")
        # Create DALI pipeline for loading video frames
        pipe = VideoFrameDaliDataloader(vids, sequence_length=128, batch_size=16, num_threads=1)
        video_iterator = DALIGenericIterator(
                [pipe],
                ['frames', 'vid', 'fid'],
                last_batch_policy=LastBatchPolicy.PARTIAL,
                # Required or iterator loops indefinitely (https://github.com/NVIDIA/DALI/issues/2873)
                # reader_name must match name in frame::VideoFrameDaliDataloader::create_pipeline.
                reader_name='reader'
            )

        udf_to_df_map = defaultdict(list)
        udf_to_pred_map = defaultdict(list)

        logger.info("executing on-the-fly UDFs")
        loading_time = 0
        transform_time = 0
        udf_execution_time = 0
        execution_time = 0
        _start = time.time()
        for batch in tqdm(video_iterator, file=sys.stdout, desc="load frames and materialize UDFs"):
            loading_time += time.time() - _start
            _start = time.time()
            batch = batch[0]
            _B, _T, _H, _W, _C = batch['frames'].shape
            # logger.debug(f"batch['frames'].shape: {_B, _T, _H, _W, _C}")
            frames = batch['frames'].reshape(-1, _H, _W, _C) # Shape: (B', H, W, C)
            non_zero_mask = frames.sum(dim=(1, 2, 3)) != 0
            frames = frames[non_zero_mask]
            vids = torch.repeat_interleave(batch['vid'], _T)[non_zero_mask].tolist()
            fids = (batch['fid'][:, None] + torch.arange(_T).to(self.device)).flatten()[non_zero_mask].tolist()
            # logger.debug(f"frames.shape: {frames.shape}, len(vids): {len(vids)}, len(fids): {len(fids)}")
            frames = frames.cpu().numpy()
            transform_time += time.time() - _start
            _start = time.time()
            for udf_name, (udf_obj, n_obj) in udf_map.items():
                for i in range(len(vids)):
                    df_grouped = df_one_object_grouped if n_obj == 1 else df_two_objects_grouped
                    if (vids[i], fids[i]) not in df_grouped.groups:
                        continue
                    df = df_grouped.get_group((vids[i], fids[i]))
                    # Execute UDF and append results
                    # NOTE: Due to data noise, multiple objects can have the same oid
                    frames_broadcast = np.broadcast_to(frames[i], (len(df), *frames[i].shape))
                    # logger.debug(f"frames[i].shape: {frames[i].shape}, frames_broadcast.shape: {frames_broadcast.shape}, df.shape: {df.shape}")
                    _udf_exec_start = time.time()
                    if n_obj == 1:
                        udf_to_pred_map[udf_name].extend(list(self.executor.map(udf_obj, frames_broadcast, df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["height"], df["width"])))
                        # df = df[["vid", "fid", "o1_oid", "pred"]]
                    elif n_obj == 2:
                        udf_to_pred_map[udf_name].extend(list(self.executor.map(udf_obj, frames_broadcast, df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["o2_oname"], df["o2_x1"], df["o2_y1"], df["o2_x2"], df["o2_y2"], df["o2_anames"], df["o1_o2_rnames"], df["o2_o1_rnames"], df["height"], df["width"])))
                        # df = df[["vid", "fid", "o1_oid", "o2_oid", "pred"]]
                    udf_execution_time += time.time() - _udf_exec_start
                    udf_to_df_map[udf_name].append(df)
            execution_time += time.time() - _start
            _start = time.time()
        logger.info(f"loading_time: {loading_time}")
        logger.info(f"transform_time: {transform_time}")
        logger.info(f"udf_execution_time: {udf_execution_time}")
        logger.info(f"execution_time: {execution_time}")
        # Concatenate and store materialized UDFs
        for udf_name, dfs in udf_to_df_map.items():
            n_obj = udf_map[udf_name][1]
            df = pd.concat(dfs, ignore_index=True)
            df["pred"] = udf_to_pred_map[udf_name]
            if n_obj == 1:
                df = df[["vid", "fid", "o1_oid", "pred"]]
            elif n_obj == 2:
                df = df[["vid", "fid", "o1_oid", "o2_oid", "pred"]]
            self.materialized_udfs[udf_name] = df
            self.materialized_udf_names.append(udf_name)
            logger.info(f"Materialized UDF: {udf_name}, shape: {df.shape}")

        self.on_the_fly_udf_names = []
        logger.info("Finish materializing on-the-fly UDFs")
        logger.info(f"available_udf_names: {self.available_udf_names}")
        logger.info(f"materialized_udf_names: {self.materialized_udf_names}")
        logger.info(f"on_the_fly_udf_names: {self.on_the_fly_udf_names}")