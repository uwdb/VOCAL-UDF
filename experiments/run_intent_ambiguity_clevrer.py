import yaml
import random
import json
import logging
import numpy as np
import argparse
import os
import sys
import ast
from openai import OpenAI, AsyncOpenAI
from collections import defaultdict
import asyncio
import importlib.util
import pandas as pd
import time
from sklearn.metrics import f1_score
from functools import partial
from tqdm import tqdm
import importlib
from itertools import chain
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch
from vocaludf.query_executor import CityFlowImageDataset, ClevrerDaliDataloader, CharadesDaliDataloader
from vocaludf.async_udf_generator import UDFGenerator
from vocaludf.udf_selector import UDFSelector, SamplingStrategy
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook, get_active_domain, transform_function, SharedResources, UDFCandidate

client = OpenAI()

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

class IntentAmbiguityUDFSelector(UDFSelector):
    def select(self, gt_udf, udf_candidate_list):
        if len(udf_candidate_list) == 0:
            return None
        udf_signature = udf_candidate_list[0].udf_signature
        udf_name, udf_vars = parse_signature(udf_signature)
        n_obj = len(udf_vars)
        # Add a dummy UDF that always returns True
        udf_description = udf_candidate_list[0].udf_description
        dummy_udf = UDFCandidate(id='dummy', payload={'udf_name': udf_name, 'udf_signature': udf_signature, 'udf_description': udf_description, 'semantic_interpretation': 'dummy', 'function_implementation': f'def py_{udf_name}(*args, **kwargs):\n    return True\n'})
        udf_candidate_list.append(dummy_udf)

        # Construct training data and test data
        df_train, df_test = self.construct_train_and_test_data(n_obj, self.n_train, self.n_test, df_with_img_column=self.program_with_pixels)

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
            y_true = self.exec_udf_with_data(df_train.iloc[labeled_index], gt_udf, {}, n_obj, requires_no_error=False, timeout=max(60, len(df_train.iloc[labeled_index])*0.2), with_pixels=False)
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
                        gt_udf,
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
                        gt_udf,
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

        self.compute_test_f1_score(gt_udf, selected_udf_candidate, udf_name, n_obj)


    def compute_test_f1_score(self, gt_udf, selected_udf_candidate, udf_name, n_obj):
        # Compute F1 score on the test set
        test_df = self.init_table_test(n_obj)
        if self.program_with_pixels:
            test_score = self.materialize_on_the_fly_udfs(gt_udf, selected_udf_candidate, udf_name, n_obj, test_df)
        else:
            test_score = self.compute_udf_score(
                gt_udf,
                selected_udf_candidate,
                udf_name,
                n_obj,
                test_df,
            )
        logger.info(f"test F1 score: {test_score}")

    def compute_udf_score(
        self,
        gt_udf,
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
        y_true = self.exec_udf_with_data(df, gt_udf, {}, n_obj, requires_no_error=False, timeout=max(60, len(df)*0.2), with_pixels=False)
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
            y_true_new = self.exec_udf_with_data(df_newly_labeled, gt_udf, {}, n_obj, requires_no_error=False, timeout=max(60, len(df_newly_labeled)*0.2), with_pixels=False)
            num_misclassified = np.sum(np.array(y_true_new != y_pred_new) * 1)
            return score, num_misclassified
        else:
            return score

    def exec_udf_with_data(self, df, udf_obj, kwargs, n_obj, requires_no_error=True, timeout=None, with_pixels=None):
        def safe_udf(udf, *args, **kwargs):
            try:
                return bool(udf(*args, **kwargs))
            except Exception as e:
                logger.exception(f"exec_udf_with_data Error: {e}")
                return False  # Default value in case of error

        if with_pixels is None:
            with_pixels = self.program_with_pixels
        else:
            with_pixels = with_pixels

        if requires_no_error:
            func = partial(udf_obj, **kwargs)
        else:
            func = partial(safe_udf, udf_obj, **kwargs)

        if n_obj == 1:
            if with_pixels:
                args = (df["img"], df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["height"], df["width"])
            else:
                args = (df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["height"], df["width"])
        elif n_obj == 2:
            if with_pixels:
                args = (df["img"], df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["o2_oname"], df["o2_x1"], df["o2_y1"], df["o2_x2"], df["o2_y2"], df["o2_anames"], df["o1_o2_rnames"], df["o2_o1_rnames"], df["height"], df["width"])
            else:
                args = (df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["o2_oname"], df["o2_x1"], df["o2_y1"], df["o2_x2"], df["o2_y2"], df["o2_anames"], df["o1_o2_rnames"], df["o2_o1_rnames"], df["height"], df["width"])

        result = list(tqdm(self.executor.map(func, *args, timeout=timeout), total=len(df), file=sys.stdout, desc="exec_udf_with_data"))

        return result

    def init_table_test(self, n_obj):
        metadata_join_clause = '' if self.dataset in ['clevr', 'clevrer'] else f'LEFT OUTER JOIN {self.dataset}_metadata m ON o1.vid = m.vid AND o1.fid = m.fid'
        height_width_clause = '320 AS height, 480 AS width' if self.dataset in ['clevr', 'clevrer'] else 'm.height AS height, m.width AS width'
        group_by_clause = 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2' if self.dataset in ['clevr', 'clevrer'] else 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2, m.height, m.width'

        attr_parameters = ','.join('?' for _ in self.attribute_domain)

        if n_obj == 1:
            if self.dataset == 'clevrer':
                where_clause = f"WHERE o1.vid >= 5000 AND o1.vid < 10000"
            elif self.dataset == 'charades':
                where_clause = f"WHERE o1.vid >= 4800 AND o1.vid < 9601"
            elif self.dataset == 'cityflow':
                where_clause = f"WHERE o1.vid >= 824 AND o1.vid < 1648"
            else:
                where_clause = ""
            sql = f"""
                SELECT
                    o1.vid AS vid, o1.fid AS fid, o1.oid AS o1_oid, o1.oname AS o1_oname,
                    o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                    COALESCE(ARRAY_AGG(DISTINCT a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
                    {height_width_clause}
                FROM {self.dataset}_objects o1
                LEFT OUTER JOIN {self.dataset}_attribute_predictions a ON o1.vid = a.vid AND o1.fid = a.fid AND o1.oid = a.oid
                {metadata_join_clause}
                {where_clause}
                GROUP BY {group_by_clause}
            """
            logger.debug(f"Create one_object table:\n{sql}")
            test_df = self.conn.execute(sql, self.attribute_domain).df()
        else:
            rel_parameters = ','.join('?' for _ in self.relationship_domain)
            if self.dataset == 'clevrer':
                attr_where_clause = f"WHERE aname = ANY([{attr_parameters}]) AND vid >= 5000 AND vid < 10000"
                obj_where_clause = "WHERE vid >= 5000 AND vid < 10000"
                rel_where_clause = f"WHERE rname = ANY([{rel_parameters}]) AND vid >= 5000 AND vid < 10000"
            elif self.dataset == 'charades':
                attr_where_clause = f"WHERE aname = ANY([{attr_parameters}]) AND vid >= 4800 AND vid < 9601"
                obj_where_clause = "WHERE vid >= 4800 AND vid < 9601"
                rel_where_clause = f"WHERE rname = ANY([{rel_parameters}]) AND vid >= 4800 AND vid < 9601"
            elif self.dataset == 'cityflow':
                attr_where_clause = f"WHERE aname = ANY([{attr_parameters}]) AND vid >= 824 AND vid < 1648"
                obj_where_clause = "WHERE vid >= 824 AND vid < 1648"
                rel_where_clause = f"WHERE rname = ANY([{rel_parameters}]) AND vid >= 824 AND vid < 1648"
            else:
                attr_where_clause = f"WHERE aname = ANY([{attr_parameters}])"
                obj_where_clause = ""
                rel_where_clause = f"WHERE rname = ANY([{rel_parameters}])"
            sql = f"""
                WITH
                    filtered_objects AS (
                        SELECT vid, fid, oid, oname, x1, y1, x2, y2
                        FROM {self.dataset}_objects
                        {obj_where_clause}
                    ),
                    filtered_attributes AS (
                        SELECT vid, fid, oid, aname
                        FROM {self.dataset}_attribute_predictions
                        {attr_where_clause}
                    ),
                    obj_with_attrs AS (
                        SELECT
                            o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                            COALESCE(ARRAY_AGG(DISTINCT a.aname), ARRAY[]::varchar[]) AS attributes
                        FROM filtered_objects o
                        LEFT OUTER JOIN filtered_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
                        GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
                    ),
                    relationships_expanded AS (
                        SELECT
                            vid, fid, oid1, oid2,
                            COALESCE(ARRAY_AGG(DISTINCT rname), ARRAY[]::varchar[]) AS rnames
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
                    {height_width_clause}
                FROM obj_with_attrs o1
                JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid <> o2.oid
                LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
                LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
                {metadata_join_clause}
            """
            logger.debug(f"Create two_objects table:\n{sql}")
            test_df = self.conn.execute(sql, self.attribute_domain + self.relationship_domain).df()

        return test_df

    def materialize_on_the_fly_udfs(self, gt_udf, selected_udf_candidate, udf_name, n_obj, test_df):
        def safe_udf(udf, *args, **kwargs):
            try:
                return udf(*args, **kwargs)
            except Exception as e:
                logger.exception(f"exec_udf_with_data Error: {e}")
                return False  # Default value in case of error

        if self.dataset == "clevrer":
            input_vids = list(range(5000, 10000))
        elif self.dataset == "charades":
            input_vids = list(range(4800, 9601))
        elif self.dataset == "cityflow":
            input_vids = list(range(824, 1648))

        parameters = ','.join('?' for _ in input_vids)
        if n_obj == 1:
            test_df = self.conn.execute(f"""
                SELECT vid, fid, o1_oid, o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, height, width
                FROM test_df
                WHERE vid = ANY([{parameters}])
            """, input_vids).df()
        else:
            test_df = self.conn.execute(f"""
                SELECT vid, fid, o1_oid, o2_oid, o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width
                FROM test_df
                WHERE vid = ANY([{parameters}])
            """, input_vids).df()

        logger.info("grouping tables by vid and fid")
        test_df_grouped = test_df.groupby(['vid', 'fid'], as_index=True, sort=False)

        logger.info("converting dataframes to numpy arrays")
        np_test_df = test_df.values

        logger.info("grouping numpy arrays by vid and fid")
        # Construct numpy arrays for each group
        grouped_np_test_df = np.array([np_test_df[i.values, :] for _, i in test_df_grouped.groups.items()], dtype=object)

        logger.info("building lookup dictionaries")
        # Lookup dictionary, where key is (vid, fid) and value is the index in the grouped_np_one_object/grouped_np_two_objects
        group_keys_test_df = dict(zip(test_df_grouped.groups.keys(), range(len(test_df_grouped.groups))))

        logger.info(f"test_df shape: {test_df.shape}")

        py_func_name = "py_{}".format(udf_name)
        exec(selected_udf_candidate.function_implementation, globals())
        udf_obj = globals()[py_func_name]

        logger.info("building video dataloader")
        # Create DALI pipeline for loading video frames
        if self.dataset == "clevrer":
            pipe = ClevrerDaliDataloader(input_vids, sequence_length=128, batch_size=1, num_threads=1)
            video_iterator = DALIGenericIterator(
            [pipe],
            ['frames', 'vid', 'fid'],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            # Required or iterator loops indefinitely (https://github.com/NVIDIA/DALI/issues/2873)
            # reader_name must match name in frame::VideoFrameDaliDataloader::create_pipeline.
            reader_name='reader'
        )
        elif self.dataset == "charades":
            pipe = CharadesDaliDataloader(input_vids, sequence_length=128, batch_size=1, num_threads=1)
            video_iterator = DALIGenericIterator(
            [pipe],
            ['frames', 'vid', 'fid'],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            # Required or iterator loops indefinitely (https://github.com/NVIDIA/DALI/issues/2873)
            # reader_name must match name in frame::VideoFrameDaliDataloader::create_pipeline.
            reader_name='reader'
        )
        elif self.dataset == "cityflow":
            df_metadata = self.conn.execute(f"""
                SELECT vname, vid, fid
                FROM cityflow_metadata
                WHERE vid = ANY([{parameters}])
            """, input_vids).df()
            data = CityFlowImageDataset(input_vids, self.config[self.dataset]["video_frames_dir"], df_metadata)
            video_iterator = DataLoader(data, batch_size=1, shuffle=False) # batch_size must be 1 because of variable image sizes
        udf_pred = []
        udf_true = []
        udf_args = []

        logger.info("executing selected UDFs")
        loading_time = 0
        transform_time = 0
        udf_execution_time = 0
        group_by_time = 0
        frames_broadcast_time = 0
        partial_udf_time = 0
        prepare_args_time = 0
        udf_map_time = 0

        _start = time.time()
        for batch in tqdm(video_iterator, file=sys.stdout, desc="load frames and materialize UDFs"):
            loading_time += time.time() - _start
            _start = time.time()
            if self.dataset in ["gqa", "clevr", "vaw", "cityflow"]:
                frames, vids, fids = batch
                frames = frames.permute(0, 2, 3, 1).cpu().numpy() # Shape: (B, C, H, W) --> (B, H, W, C)
                vids = vids.tolist()
                fids = fids.tolist()
            else:
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

            for i in range(len(vids)):
                _start = time.time()
                grouped_idx = group_keys_test_df.get((vids[i], fids[i]), -1)
                if grouped_idx == -1:
                    group_by_time += time.time() - _start
                    continue
                arr = grouped_np_test_df[grouped_idx]

                group_by_time += time.time() - _start

                # Execute UDF and append results
                # NOTE: Due to data noise, multiple objects can have the same oid

                _start = time.time()
                func = partial(safe_udf, udf_obj)
                gt_func = partial(safe_udf, gt_udf)
                # func = py_func
                partial_udf_time += time.time() - _start

                if n_obj == 1:
                    _start = time.time()
                    # args = [arr[:, i] for i in range(3, 11)]
                    prepare_args_time += time.time() - _start
                    _start = time.time()
                    res = []
                    gt_res = []
                    for j in range(len(arr)):
                        res.append(func(frames[i], *arr[j, 3:11]))
                        gt_res.append(gt_func(*arr[j, 3:11]))
                    # res = self.executor.map(func, frames_broadcast, *args)
                    udf_execution_time += time.time() - _start
                elif n_obj == 2:
                    _start = time.time()
                    # args = [arr[0, i] for i in range(4, 20)]
                    prepare_args_time += time.time() - _start
                    _start = time.time()
                    res = []
                    gt_res = []
                    for j in range(len(arr)):
                        res.append(func(frames[i], *arr[j, 4:20]))
                        gt_res.append(gt_func(*arr[j, 4:20]))
                    # res = self.executor.map(func, frames_broadcast, *args)
                    udf_execution_time += time.time() - _start

                _start = time.time()
                udf_pred.append(res)
                udf_true.append(gt_res)
                udf_map_time += time.time() - _start

            _start = time.time()
        logger.info(f"loading_time: {loading_time}")
        logger.info(f"transform_time: {transform_time}")
        logger.info(f"group_by_time: {group_by_time}")
        logger.info(f"frames_broadcast_time: {frames_broadcast_time}")
        logger.info(f"partial_udf_time: {partial_udf_time}")
        logger.info(f"prepare_args_time: {prepare_args_time}")
        logger.info(f"udf_execution_time: {udf_execution_time}")
        logger.info(f"udf_map_time: {udf_map_time}")

        # Concatenate and store materialized UDFs
        # for udf_name, dfs in udf_to_df_map.items():
        udf_pred = list(chain(*udf_pred))
        udf_true = list(chain(*udf_true))
        score = f1_score(udf_true, udf_pred, zero_division=0.0)
        logger.info(f"Materialized UDF: {udf_name}, F1 score: {score}")
        return score

async def main():
    # VOCAL-UDF: python run_intent_ambiguity_clevrer.py --run_id 0 --dataset "clevrer" --udf_name "behind" --interpretation_id 0 --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5 --num_workers 8 --n_train_distill 100 --selection_strategy "program" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"

    # Baseline: python run_intent_ambiguity_clevrer.py --run_id 0 --dataset "clevrer" --udf_name "far" --interpretation_id 0 --budget 20 --n_selection_samples 500 --num_interpretations 1 --program_with_pixels --num_parameter_search 1 --num_workers 8 --n_train_distill 100 --selection_strategy "program" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"

    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    parser = argparse.ArgumentParser()
    # parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    # parser.add_argument("--query_filename", type=str, help="query filename")
    parser.add_argument("--udf_name", type=str, help="UDF name")
    parser.add_argument("--interpretation_id", type=int, help="interpretation id")
    parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in UDF")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument("--n_selection_samples", type=int, default=500, help="number of sampled videos per selection iteration")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--program_with_pixels", action="store_true", help="program with pixels")
    parser.add_argument("--program_with_pretrained_models", action="store_true", help="program with pretrained models")
    parser.add_argument("--num_workers", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--llm_method", type=str, choices=["gpt4v", "llava"], default="gpt4v", help="LLM method for distill model annotations")
    parser.add_argument("--is_async", action="store_true", help="use async for distilled-model UDF labeling")
    parser.add_argument("--openai_model_name", type=str, help="OpenAI model name")

    args = parser.parse_args()
    # query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    # query_filename = args.query_filename
    udf_name = args.udf_name
    interpretation_id = args.interpretation_id
    allow_kwargs_in_udf = args.allow_kwargs_in_udf
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    n_selection_samples = args.n_selection_samples
    llm_method = args.llm_method
    num_interpretations = args.num_interpretations
    program_with_pixels = args.program_with_pixels
    program_with_pretrained_models = args.program_with_pretrained_models
    if program_with_pretrained_models:
        assert program_with_pixels, "program_with_pretrained_models requires program_with_pixels"
    num_workers = args.num_workers
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    is_async = args.is_async
    openai_model_name = args.openai_model_name

    if num_interpretations == 1:
        # For the baseline, we set the temperature to a higher value to encourage variety across runs
        config["udf_generator"]["temperature"] = 1
        config["udf_generator"]["top_p"] = 1
    assert selection_strategy == "program"

    if udf_name == "behind":
        udf_signature = "behind(o0, o1)"
        udf_description = "Whether o0 is behind o1."
    elif udf_name == "near":
        udf_signature = "near(o0, o1)"
        udf_description = "Whether o0 is near o1."
    elif udf_name == "far":
        udf_signature = "far(o0, o1)"
        udf_description = "Whether o0 is far away from o1."
    elif udf_name == "location_bottom":
        udf_signature = "location_bottom(o0)"
        udf_description = "Whether o0 is at the bottom of the frame."
    else:
        raise ValueError("Invalid UDF name")

    # Dynamically import the ground truth UDF
    module_name = f"udfs.gt_{udf_name}"
    module = importlib.import_module(module_name)
    gt_udf = getattr(module, f"gt_{interpretation_id}")

    config_name = "ninterp={}-nparams={}-kwargs={}-pixels={}-pretrained_models={}-ntrain_distill={}-nselection_samples={}-selection={}-budget={}-llm_method={}".format(
        num_interpretations,
        num_parameter_search,
        allow_kwargs_in_udf,
        program_with_pixels,
        program_with_pretrained_models,
        n_train_distill,
        n_selection_samples,
        selection_strategy,
        labeling_budget,
        llm_method,
    )

    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        "intent_ambiguity",
        dataset,
        udf_name,
        config_name,
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(os.path.join(log_dir, "interpretation={}-run={}.log".format(interpretation_id, run_id)), mode="w")
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

    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )

    registered_udfs_json = json.load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r"))
    registered_functions = registered_udfs_json[f"{dataset}_base"]
    if num_interpretations == 1 and not allow_kwargs_in_udf:
        registered_functions = []
    logger.info("Registered functions: {}".format(registered_functions))
    object_domain, relationship_domain, attribute_domain = get_active_domain(config, dataset, registered_functions)
    logger.info("Active domains: object={}, relationship={}, attribute={}".format(object_domain, relationship_domain, attribute_domain))

    save_labeled_data = False
    load_labeled_data = False

    shared_resources = SharedResources(
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
        None, # query_filename,
        None, # query_id,
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
    )

    logger.info(f"UDF generation started")
    ug = UDFGenerator(
        shared_resources,
        udf_signature,
        udf_description,
        None, # gt_udf_name
        )
    udf_candidate_list, llm_positive_df, llm_negative_df = await ug.implement()
    logger.info(f"UDF generation finished")

    if num_interpretations == 1 and not allow_kwargs_in_udf:
        assert len(udf_candidate_list) == 1, "Only one UDF candidate should be generated"

    if len(udf_candidate_list) > 1:
        logger.info(f"UDF selection started")
        us = IntentAmbiguityUDFSelector(shared_resources, llm_positive_df, llm_negative_df)
        us.select(gt_udf, udf_candidate_list)
        logger.info(f"UDF selection finished")
    else:
        logger.info("Only one UDF candidate is generated. No selection is needed.")
        udf_signature = udf_candidate_list[0].udf_signature
        udf_name, udf_vars = parse_signature(udf_signature)
        n_obj = len(udf_vars)
        us = IntentAmbiguityUDFSelector(shared_resources, llm_positive_df, llm_negative_df)
        us.compute_test_f1_score(gt_udf, udf_candidate_list[0], udf_name, n_obj)

if __name__ == '__main__':
    asyncio.run(main())