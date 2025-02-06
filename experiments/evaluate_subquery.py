import yaml
import random
import json
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook, get_active_domain
import logging
import numpy as np
from vocaludf.query_executor import QueryExecutor
from vocaludf.parser import parse
from src.utils import program_to_dsl
import argparse
import os
import sys
import duckdb

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_missing_udfs", type=int, help="number of missing UDFs")
    parser.add_argument('--query_id', type=int, help='question id')
    parser.add_argument("--num_workers", type=int, default=4, help="Maximum number of tasks to execute at once")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--query_filename", type=str, help="query filename")
    parser.add_argument("--pred_batch_size", type=int, default=262144, help="batch size for prediction data loader")
    parser.add_argument("--dali_batch_size", type=int, default=16, help="batch size for DALI")
    args = parser.parse_args()
    num_missing_udfs = args.num_missing_udfs
    query_id = args.query_id
    dataset = args.dataset
    query_filename = args.query_filename
    pred_batch_size = args.pred_batch_size
    dali_batch_size = args.dali_batch_size

    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    program_with_pixels = False
    num_workers = args.num_workers

    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_filename}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    gt_dsl = input_query['dsl']
    positive_videos = input_query["positive_videos"]
    if dataset in ["gqa", "vaw"]:
        conn = duckdb.connect(
            database=os.path.join(config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )
        vids = conn.execute(f"SELECT DISTINCT vid FROM {dataset}_metadata ORDER BY vid ASC").df()["vid"].tolist()
        y_true = [1 if vid in positive_videos else 0 for vid in vids]
    else:
        y_true = [1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"])]

    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        "query_execution",
        dataset,
        query_filename,
        "num_missing_udfs={}".format(num_missing_udfs),
        "queries_unavailable_udfs_removed",
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(os.path.join(log_dir, "qid={}.log".format(query_id)), mode="w")
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

    registered_udfs_json = json.load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r"))
    if "single_semantic" in query_filename:
        registered_functions = [{
            "signature": "object(o0, name)",
            "description": "Whether o0 is an object with the given name.",
            "function_implementation": ""
        }]
    else:
        registered_functions = registered_udfs_json[f"{dataset}_base"]
        new_modules = input_query["new_modules"]
        assert num_missing_udfs >= 0 and num_missing_udfs <= len(new_modules), "num_missing_udfs must be between 0 and len(new_modules)"
        for new_module in new_modules[:(len(new_modules)-num_missing_udfs)]:
            registered_functions.append(registered_udfs_json[dataset][new_module])
    logger.info("Registered functions: {}".format(registered_functions))
    object_domain, relationship_domain, attribute_domain = get_active_domain(config, dataset, registered_functions)
    logger.info("Active domains: object={}, relationship={}, attribute={}".format(object_domain, relationship_domain, attribute_domain))
    available_udf_names = [parse_signature(func["signature"])[0] for func in registered_functions]
    materialized_df_names = []
    on_the_fly_udf_names = []
    parsed_program = parse().parseString(gt_dsl, parseAll=True).as_dict()
    # Remove the UDF from the parsed program
    new_query = []
    for sg in parsed_program['query']:
        new_scene_graph = []
        for pred in sg['scene_graph']:
            if pred['predicate'].lower() in available_udf_names:
                new_scene_graph.append(pred)
        if len(new_scene_graph) > 0:
            new_query.append({'scene_graph': new_scene_graph, 'duration_constraint': sg['duration_constraint']})
    parsed_program = {'query': new_query}
    logger.info(f"parsed_program: {parsed_program}")
    parsed_dsl = program_to_dsl(parsed_program['query'], rewrite_variables=False, sort_variables=False)
    logger.info(f"parsed_dsl: {parsed_dsl}")
    qe = QueryExecutor(config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_df_names, on_the_fly_udf_names, program_with_pixels, num_workers, pred_batch_size, dali_batch_size)
    qe.run(parsed_program, y_true, debug=False)
