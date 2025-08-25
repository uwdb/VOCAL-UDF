import yaml
import random
import json
from vocaludf.utils import parse_signature, setup_logging, duckdb_execute_video_materialize
import logging
import numpy as np
from vocaludf.query_executor import QueryExecutor, ClevrerDaliDataloader
import argparse
import os
import sys
import resource
import duckdb
from vocaludf.query_parser import QueryParser
import time
from sklearn.metrics import f1_score
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from functools import partial
from tqdm import tqdm
import torch
import pandas as pd
from collections import defaultdict
from itertools import chain

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

project_root = os.getenv("PROJECT_ROOT")

class SimplifiedQueryExecutor(QueryExecutor):
    def init_table(self):
        attr_parameters = ','.join('?' for _ in self.attribute_domain)
        sql = f"""
            CREATE TEMPORARY TABLE one_object AS
            SELECT
                o1.vid AS vid, o1.fid // 4 AS fid, o1.oid AS o1_oid, o1.oname AS o1_oname,
                o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                COALESCE(ARRAY_AGG(DISTINCT a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
                320 AS height, 480 AS width
            FROM {self.dataset}_objects o1
            LEFT OUTER JOIN {self.dataset}_attribute_predictions a ON o1.vid = a.vid AND o1.fid = a.fid AND o1.oid = a.oid
            WHERE o1.vid >= 9500 AND o1.vid < 10000 AND o1.fid % 4 = 0
            GROUP BY o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2
        """
        logger.debug(f"Create one_object table:\n{sql}")
        self.conn.execute(sql, self.attribute_domain)

        rel_parameters = ','.join('?' for _ in self.relationship_domain)

        sql = f"""
            CREATE TEMPORARY TABLE two_objects AS
            WITH
                filtered_objects AS (
                    SELECT vid, fid, oid, oname, x1, y1, x2, y2
                    FROM {self.dataset}_objects
                    WHERE vid >= 9500 AND vid < 10000
                ),
                filtered_attributes AS (
                    SELECT vid, fid, oid, aname
                    FROM {self.dataset}_attribute_predictions
                    WHERE aname = ANY([{attr_parameters}]) AND vid >= 9500 AND vid < 10000
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
                    WHERE rname = ANY([{rel_parameters}]) AND vid >= 9500 AND vid < 10000
                    GROUP BY vid, fid, oid1, oid2
                )
            SELECT
                o1.vid AS vid, o1.fid // 4 AS fid,
                o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2, o1.attributes AS o1_anames,
                o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2, o2.attributes AS o2_anames,
                COALESCE(r1.rnames, ARRAY[]::varchar[]) AS o1_o2_rnames,
                COALESCE(r2.rnames, ARRAY[]::varchar[]) AS o2_o1_rnames,
                320 AS height, 480 AS width
            FROM obj_with_attrs o1
            JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid <> o2.oid
            LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
            LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
            WHERE o1.fid % 4 = 0
        """
        logger.debug(f"Create two_objects table:\n{sql}")
        self.conn.execute(sql, self.attribute_domain + self.relationship_domain)

    def get_materialized_df(self, func):
        df_with_pred = super().get_materialized_df(func)
        # select rows with fid % 4 == 0, and then set fid = fid // 4
        df_with_pred = df_with_pred[df_with_pred["fid"] % 4 == 0]
        df_with_pred.loc[:, "fid"] = df_with_pred["fid"] // 4
        return df_with_pred

    def run(self, program, y_true, debug=False):
        _start_run = time.time()
        # Bind each dataframe to a variable name, so that we can refer to them in the SQL query
        for udf_name, df in self.materialized_udfs.items():
            exec(f"{udf_name} = df")
        exec_func = duckdb_execute_video_materialize
        input_vids = list(range(9500, 10000))
        logger.info("Running query: {}".format(program["query"]))

        if self.program_with_pixels:
            _start = time.time()
            # First, execute the query with all on-the-fly UDFs removed
            query_program = self.remove_on_the_fly_udfs(program["query"])
            result = exec_func(
                self.conn,
                query_program,
                None,
                available_udf_names=self.available_udf_names,
                materialized_udf_names=self.materialized_udf_names,
                on_the_fly_udf_names=[]
            )
            logger.info("Time to execute first query: {}".format(time.time() - _start))
            result = sorted(result)
            logger.info("output vids: {}".format(result))
            if len(self.on_the_fly_udf_names) > 0 and len(result) > 0:
                self.materialize_on_the_fly_udfs(result)
                for udf_name, df in self.materialized_udfs.items():
                    exec(f"{udf_name} = df")
                _start = time.time()
                # Then, execute the query with all on-the-fly UDFs over the result of the previous query
                result = exec_func(
                    self.conn,
                    program["query"],
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
            result = exec_func(
                self.conn,
                program["query"],
                None,
                available_udf_names=self.available_udf_names,
                materialized_udf_names=self.materialized_udf_names,
                on_the_fly_udf_names=self.on_the_fly_udf_names
            )
            logger.info("Time to execute query: {}".format(time.time() - _start))
            result = sorted(result)
            logger.info("output vids: {}".format(result))
        # logger.info("output vids: {}".format(result))
        y_pred = [1 if vid in result else 0 for vid in input_vids]

        f1 = f1_score(y_true[:len(y_pred)], y_pred)
        self.query_execution_time = time.time() - _start_run
        logger.info("Test data initialization time: {}".format(self.test_data_init_time))
        logger.info("Query execution time: {}".format(self.query_execution_time))
        logger.info("F1 score: {}".format(f1))
        return y_pred

    def materialize_on_the_fly_udfs(self, vids):
        def safe_udf(udf, *args, **kwargs):
            try:
                return udf(*args, **kwargs)
            except Exception as e:
                logger.exception(f"exec_udf_with_data Error: {e}")
                return False  # Default value in case of error

        logger.info("Start materializing on-the-fly UDFs")

        # Filter on-the-fly UDFs
        on_the_fly_udfs = [func for func in self.registered_functions if parse_signature(func["signature"])[0] in self.on_the_fly_udf_names]

        logger.info("filtering tables by matching vids")
        # Group one_object and two_objects tables by vid and fid
        parameters = ','.join('?' for _ in vids)
        df_one_object = self.conn.execute(f"""
            SELECT vid, fid, o1_oid, o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, height, width
            FROM one_object
            WHERE vid = ANY([{parameters}])
        """, vids).df()

        df_two_objects = self.conn.execute(f"""
            SELECT vid, fid, o1_oid, o2_oid, o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width
            FROM two_objects
            WHERE vid = ANY([{parameters}])
        """, vids).df()

        logger.info("grouping tables by vid and fid")
        df_one_object_grouped = df_one_object.groupby(['vid', 'fid'], as_index=True, sort=False)
        df_two_objects_grouped = df_two_objects.groupby(['vid', 'fid'], as_index=True, sort=False)

        logger.info("converting dataframes to numpy arrays")
        np_one_object = df_one_object.values
        np_two_objects = df_two_objects.values

        logger.info("grouping numpy arrays by vid and fid")
        # Construct numpy arrays for each group
        grouped_np_one_object = np.array([np_one_object[i.values, :] for _, i in df_one_object_grouped.groups.items()], dtype=object)
        grouped_np_two_objects = np.array([np_two_objects[i.values, :] for _, i in df_two_objects_grouped.groups.items()], dtype=object)

        logger.info("building lookup dictionaries")
        # Lookup dictionary, where key is (vid, fid) and value is the index in the grouped_np_one_object/grouped_np_two_objects
        group_keys_one_object = dict(zip(df_one_object_grouped.groups.keys(), range(len(df_one_object_grouped.groups))))
        group_keys_two_objects = dict(zip(df_two_objects_grouped.groups.keys(), range(len(df_two_objects_grouped.groups))))

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
        pipe = ClevrerDaliDataloader(vids, sequence_length=128, batch_size=self.dali_batch_size, num_threads=1)
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
        group_by_time = 0
        frames_broadcast_time = 0
        partial_udf_time = 0
        prepare_args_time = 0
        udf_map_time = 0

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

            for udf_name, (udf_obj, n_obj) in udf_map.items():
                for i in range(len(vids)):
                    if fids[i] % 4 != 0:
                        continue
                    _start = time.time()
                    grouped_idx = group_keys_one_object.get((vids[i], fids[i] // 4), -1) if n_obj == 1 else group_keys_two_objects.get((vids[i], fids[i] // 4), -1)
                    if grouped_idx == -1:
                        group_by_time += time.time() - _start
                        continue
                    arr = grouped_np_one_object[grouped_idx] if n_obj == 1 else grouped_np_two_objects[grouped_idx]

                    group_by_time += time.time() - _start

                    # Execute UDF and append results
                    # NOTE: Due to data noise, multiple objects can have the same oid

                    _start = time.time()
                    func = partial(safe_udf, udf_obj)
                    # func = py_func
                    partial_udf_time += time.time() - _start

                    if n_obj == 1:
                        _start = time.time()
                        # args = [arr[:, i] for i in range(3, 11)]
                        prepare_args_time += time.time() - _start
                        _start = time.time()
                        res = []
                        for j in range(len(arr)):
                            res.append(func(frames[i], *arr[j, 3:11]))
                        # res = self.executor.map(func, frames_broadcast, *args)
                        udf_execution_time += time.time() - _start
                    elif n_obj == 2:
                        _start = time.time()
                        # args = [arr[0, i] for i in range(4, 20)]
                        prepare_args_time += time.time() - _start
                        _start = time.time()
                        res = []
                        for j in range(len(arr)):
                            res.append(func(frames[i], *arr[j, 4:20]))
                        # res = self.executor.map(func, frames_broadcast, *args)
                        udf_execution_time += time.time() - _start

                    _start = time.time()
                    udf_to_pred_map[udf_name].append(res)
                    udf_to_df_map[udf_name].append(arr)
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
        for udf_name, dfs in udf_to_df_map.items():
            n_obj = udf_map[udf_name][1]
            arr = np.concatenate(dfs, axis=0)
            df = pd.DataFrame(arr, columns=["vid", "fid", "o1_oid", "o1_oname", "o1_x1", "o1_y1", "o1_x2", "o1_y2", "o1_anames", "height", "width"] if n_obj == 1 else ["vid", "fid", "o1_oid", "o2_oid", "o1_oname", "o1_x1", "o1_y1", "o1_x2", "o1_y2", "o1_anames", "o2_oname", "o2_x1", "o2_y1", "o2_x2", "o2_y2", "o2_anames", "o1_o2_rnames", "o2_o1_rnames", "height", "width"])
            df["pred"] = list(chain(*udf_to_pred_map[udf_name]))
            if n_obj == 1:
                df = df[["vid", "fid", "o1_oid", "pred"]]
            elif n_obj == 2:
                df = df[["vid", "fid", "o1_oid", "o2_oid", "pred"]]
            self.materialized_udfs[f"udf_{udf_name}"] = df
            self.materialized_udf_names.append(f"udf_{udf_name}")
            logger.info(f"Materialized UDF: {udf_name}, shape: {df.shape}")

        self.on_the_fly_udf_names = []
        logger.info("Finish materializing on-the-fly UDFs")
        logger.info(f"available_udf_names: {self.available_udf_names}")
        logger.info(f"materialized_udf_names: {self.materialized_udf_names}")
        logger.info(f"on_the_fly_udf_names: {self.on_the_fly_udf_names}")

if __name__ == '__main__':
    # python run_clevrer_simplified_query.py --num_missing_udfs 3 --query_id 2 --run_id 0 --dataset "clevrer" --query_filename "simplified_3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5 --num_workers 8 --n_train_distill 100 --selection_strategy "both" --pred_batch_size 4096 --dali_batch_size 1 --llm_method "gpt4v" --openai_model_name "gpt-4o"
    config = yaml.safe_load(
        open(os.path.join(project_root, "configs", "config.yaml"), "r")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_missing_udfs", type=int, help="number of missing UDFs")
    parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--query_filename", type=str, help="query filename")
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
    parser.add_argument("--pred_batch_size", type=int, default=262144, help="batch size for prediction data loader")
    parser.add_argument("--dali_batch_size", type=int, default=16, help="batch size for DALI")
    parser.add_argument("--llm_method", type=str, choices=["gpt", "llava"], default="gpt", help="LLM method for distill model annotations")
    parser.add_argument("--openai_model_name", type=str, help="OpenAI model name")

    args = parser.parse_args()
    num_missing_udfs = args.num_missing_udfs
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    assert dataset == 'clevrer', "Only Clevrer dataset is supported"
    query_filename = args.query_filename
    allow_kwargs_in_udf = args.allow_kwargs_in_udf
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    n_selection_samples = args.n_selection_samples
    llm_method = args.llm_method
    openai_model_name = args.openai_model_name
    num_interpretations = args.num_interpretations
    program_with_pixels = args.program_with_pixels
    program_with_pretrained_models = args.program_with_pretrained_models
    if program_with_pretrained_models:
        assert program_with_pixels, "program_with_pretrained_models requires program_with_pixels"
    num_workers = args.num_workers
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    pred_batch_size = args.pred_batch_size
    dali_batch_size = args.dali_batch_size

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

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_filename}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    user_query = input_query["question"]
    positive_videos = input_query["positive_videos"]
    y_true = [1 if i in positive_videos else 0 for i in range(9500, 10000)]


    # Set up logging
    base_dir = os.path.join(
        "query_execution",
        dataset,
        query_filename,
        "num_missing_udfs={}".format(num_missing_udfs),
        config_name,
    )
    log_filename = "qid={}-run={}.log".format(query_id, run_id)
    setup_logging(config, base_dir, log_filename, logger)

    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )

    with open(
        os.path.join(
            config["output_dir"],
            "udf_generation",
            dataset,
            query_filename.replace("simplified_", ""),
            "num_missing_udfs={}".format(num_missing_udfs),
            config_name,
            "qid={}-run={}.json".format(query_id, run_id),
        ),
        "r",
    ) as f:
        generation_output = json.load(f)
    program_with_pixels = generation_output["program_with_pixels"]
    # parsed_program = generation_output["parsed_program"]
    object_domain = generation_output["object_domain"]
    relationship_domain = generation_output["relationship_domain"]
    attribute_domain = generation_output["attribute_domain"]
    registered_functions = generation_output["registered_functions"]
    available_udf_names = generation_output["available_udf_names"]
    materialized_df_names = generation_output["materialized_df_names"]
    on_the_fly_udf_names = generation_output["on_the_fly_udf_names"]

    # Step 1: Parse the query using the available UDFs
    logger.info("Query parsing started")
    qp = QueryParser(
        config,
        prompt_config,
        dataset,
        registered_functions,
        object_domain,
        run_id,
        openai_model_name,
        allow_new_udfs=False,
    )
    qp.parse(user_query)
    parsed_program = qp.get_parsed_program()
    parsed_dsl = qp.get_parsed_query()
    logger.info("parsed_program: {}".format(parsed_program))
    logger.info("Query parsing finished")

    # Step 2: Execute the query
    logger.info("Query execution started")
    qe = SimplifiedQueryExecutor(config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_df_names, on_the_fly_udf_names, program_with_pixels, num_workers, pred_batch_size, dali_batch_size)
    qe.run(parsed_program, y_true, debug=False)
    logger.info("Query execution finished")

    logger.info("Peak memory usuage (in GB): {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0))