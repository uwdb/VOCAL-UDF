import yaml
import random
import json
import logging
import numpy as np
import argparse
import os
import resource
from vocaludf.query_executor import QueryExecutor
from vocaludf.utils import setup_logging

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    # python run_query_executor.py --num_missing_udfs 3 --query_id 2 --run_id 3 --dataset "clevrer" --query_class_name "3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5 --cpus 8 --n_train_distill 100 --selection_strategy "both" --selection_labels "user" --pred_batch_size 4096 --dali_batch_size 1 --llm_method "gpt4v"
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_missing_udfs", type=int, help="number of missing UDFs")
    parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--query_class_name", type=str, help="query class name")
    parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in UDF")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument("--n_selection_samples", type=int, default=500, help="number of sampled videos per selection iteration")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--program_with_pixels", action="store_true", help="program with pixels")
    parser.add_argument("--program_with_pretrained_models", action="store_true", help="program with pretrained models")
    parser.add_argument("--cpus", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--selection_labels", type=str, choices=["none", "user", "llm"], default="user", help="strategy for UDF selection")
    parser.add_argument("--pred_batch_size", type=int, default=262144, help="batch size for prediction data loader")
    parser.add_argument("--dali_batch_size", type=int, default=16, help="batch size for DALI")
    parser.add_argument("--llm_method", type=str, choices=["gpt4v", "llava"], default="gpt4v", help="LLM method for distill model annotations")
    parser.add_argument("--udf_selection_mode", type=str, choices=["random", "active", "no_dummy"], default="active", help="UDF selection mode")

    args = parser.parse_args()
    num_missing_udfs = args.num_missing_udfs
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    query_class_name = args.query_class_name
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
    num_workers = args.cpus
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    selection_labels = args.selection_labels
    # if selection_strategy != "program":
    #     assert program_with_pixels, "selection_strategy != 'program' requires program_with_pixels"

    if selection_strategy == "both":
        assert selection_labels != "none"
    elif selection_strategy == "model":
        assert selection_labels == "none"
    pred_batch_size = args.pred_batch_size
    dali_batch_size = args.dali_batch_size
    udf_selection_mode = args.udf_selection_mode

    config_name = "ninterp={}-nparams={}-kwargs={}-pixels={}-pretrained_models={}-ntrain_distill={}-nselection_samples={}-selection={}-labels={}-budget={}-llm_method={}".format(
        num_interpretations,
        num_parameter_search,
        allow_kwargs_in_udf,
        program_with_pixels,
        program_with_pretrained_models,
        n_train_distill,
        n_selection_samples,
        selection_strategy,
        selection_labels,
        labeling_budget,
        llm_method,
    )
    if udf_selection_mode != "active":
        config_name = f"{udf_selection_mode}_{config_name}"

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_class_name}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    positive_videos = input_query["positive_videos"]
    y_true = [1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"] // 2, config[dataset]["dataset_size"])]

    # Set up logging
    base_dir = os.path.join(
        "query_execution",
        dataset,
        query_class_name,
        "num_missing_udfs={}".format(num_missing_udfs),
        config_name,
    )
    log_filename = "qid={}-run={}.log".format(query_id, run_id)
    setup_logging(config, base_dir, log_filename, logger)

    with open(
        os.path.join(
            config["output_dir"],
            "udf_generation",
            dataset,
            query_class_name,
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

    logger.info("parsed_program: {}".format(parsed_program))

    qe = QueryExecutor(config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_df_names, on_the_fly_udf_names, program_with_pixels, num_workers, pred_batch_size, dali_batch_size)
    qe.run(parsed_program, y_true, debug=False)

    logger.info("Peak memory usuage (in GB): {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0))