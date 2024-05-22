import yaml
import random
import json
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook, get_active_domain
import logging
import numpy as np
from vocaludf.query_executor import QueryExecutor
import argparse
import os
import sys
import resource

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_missing_udfs", type=int, help="number of missing UDFs")
    parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in UDF")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--program_with_pixels", action="store_true", help="program with pixels")
    parser.add_argument("--program_with_pretrained_models", action="store_true", help="program with pretrained models")
    parser.add_argument("--cpus", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--selection_labels", type=str, choices=["none", "user", "llm"], default="user", help="strategy for UDF selection")
    parser.add_argument("--pred_batch_size", type=int, default=262144, help="batch size for prediction data loader")
    parser.add_argument("--dali_batch_size", type=int, default=16, help="batch size for DALI")

    args = parser.parse_args()
    num_missing_udfs = args.num_missing_udfs
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    allow_kwargs_in_udf = args.allow_kwargs_in_udf
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    num_interpretations = args.num_interpretations
    program_with_pixels = args.program_with_pixels
    program_with_pretrained_models = args.program_with_pretrained_models
    if program_with_pretrained_models:
        assert program_with_pixels, "program_with_pretrained_models requires program_with_pixels"
    num_workers = args.cpus
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    selection_labels = args.selection_labels
    if selection_strategy != "program":
        assert program_with_pixels, "selection_strategy != 'program' requires program_with_pixels"

    if selection_strategy == "both":
        assert selection_labels != "none"
    elif selection_strategy == "model":
        assert selection_labels == "none"
    pred_batch_size = args.pred_batch_size
    dali_batch_size = args.dali_batch_size

    config_name = "ninterp={}-nparams={}-kwargs={}-pixels={}-pretrained_models={}-ntrain_distill={}-selection={}-labels={}-budget={}".format(
        num_interpretations,
        num_parameter_search,
        allow_kwargs_in_udf,
        program_with_pixels,
        program_with_pretrained_models,
        n_train_distill,
        selection_strategy,
        selection_labels,
        labeling_budget,
    )

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = config[dataset]["input_query_file"]
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    positive_videos = input_query["positive_videos"]
    y_true = [1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"])]

    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        "query_execution",
        dataset,
        "num_missing_udfs={}".format(num_missing_udfs),
        config_name,
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(os.path.join(log_dir, "qid={}-run={}.log".format(query_id, run_id)), mode="w")
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

    with open(
        os.path.join(
            config["output_dir"],
            "udf_generation",
            dataset,
            "num_missing_udfs={}".format(num_missing_udfs),
            config_name,
            "qid={}-run={}.json".format(query_id, run_id),
        ),
        "r",
    ) as f:
        generation_output = json.load(f)
    program_with_pixels = generation_output["program_with_pixels"]
    parsed_program = generation_output["parsed_program"]
    object_domain = generation_output["object_domain"]
    relationship_domain = generation_output["relationship_domain"]
    attribute_domain = generation_output["attribute_domain"]
    registered_functions = generation_output["registered_functions"]
    available_udf_names = generation_output["available_udf_names"]
    materialized_df_names = generation_output["materialized_df_names"]
    on_the_fly_udf_names = generation_output["on_the_fly_udf_names"]

    qe = QueryExecutor(config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_df_names, on_the_fly_udf_names, program_with_pixels, num_workers, pred_batch_size, dali_batch_size)
    qe.run(parsed_program, y_true, debug=False)

    logger.info("Peak memory usuage (in GB): {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0))