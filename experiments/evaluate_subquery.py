import yaml
import random
import json
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook
import logging
import numpy as np
from vocaludf.query_executor import QueryExecutor
from vocaludf.parser import parse
from src.utils import program_to_dsl
import argparse
import os
import sys

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_id', type=int, help='question id')
    parser.add_argument("--cpus", type=int, default=4, help="Maximum number of tasks to execute at once")
    parser.add_argument("--dataset", type=str, help="dataset name")
    args = parser.parse_args()
    query_id = args.query_id
    dataset = args.dataset

    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    program_with_pixels = False
    num_workers = args.cpus

    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        "udf_generation",
        dataset,
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

    # parsed_program = {'query': [{'scene_graph': [{'predicate': 'left_of', 'variables': ['o2', 'o0']}, {'predicate': 'color_red', 'variables': ['o0']}, {'predicate': 'front_of', 'variables': ['o1', 'o2']}], 'duration_constraint': 15}, {'scene_graph': [{'predicate': 'far_from', 'variables': ['o1', 'o2']}, {'predicate': 'left_of', 'variables': ['o0', 'o2']}], 'duration_constraint': 5}, {'scene_graph': [{'predicate': 'behind_of', 'variables': ['o2', 'o0']}, {'predicate': 'material_metal', 'variables': ['o1']}], 'duration_constraint': 1}]}
    input_query_file = config[dataset]["input_query_file"]
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    gt_dsl = input_query['dsl']
    positive_videos = input_query["positive_videos"]
    y_true = [1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"])]

    # Available UDFs
    object_domain = ['object']
    relationship_domain = ['left_of', 'front_of']
    attribute_domain = ['location_left', 'location_top', 'color_gray', 'color_red', 'color_blue', 'color_green', 'shape_cube', 'shape_sphere', 'material_rubber', 'shape_cylinder']

    registered_functions = json.load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r")
    )["test"]
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
    qe = QueryExecutor(config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_df_names, on_the_fly_udf_names, program_with_pixels, num_workers)
    qe.run(parsed_program, y_true, debug=False)