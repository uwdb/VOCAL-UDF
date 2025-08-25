from collections import defaultdict
import time
import pandas as pd
import yaml
import random
import json
import os
from vocaludf.utils import parse_signature, setup_logging, get_active_domain, SharedResources
import logging
import argparse
from sentence_transformers import SentenceTransformer, util
import numpy as np
from vocaludf.query_parser import QueryParser
from vocaludf.udf_selector import UDFSelector
from vocaludf.query_executor import QueryExecutor
import duckdb
import sys
import resource
import asyncio
import ast
import re
import copy

# logging.basicConfig()
logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

project_root = os.getenv("PROJECT_ROOT")

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s GB'''%(point, usage/1024.0/1024.0 )

class RandomUDFSelector(UDFSelector):
    def random_count(self, gt_udf_name, n_obj, pos_count, neg_count):
        # Construct training data and test data
        df_train, df_test = self.construct_train_and_test_data(n_obj, self.n_train, self.n_test, df_with_img_column=False)

        # Select new video segments to label
        # TODO: df_train and llm_positive_df may contain the same tuples
        labeled_index = []

        iter = 0
        y_true = None
        while (y_true is None) or sum(y_true) < 10:
            logger.info("iter {}: ".format(iter))

            unlabeled_index = np.setdiff1d(
                np.arange(len(df_train)), labeled_index, assume_unique=True
            )
            logger.debug("len(unlabeled_index): {}".format(len(unlabeled_index)))

            # Randomly select one
            new_labeled_index = [int(np.random.choice(unlabeled_index, 1, replace=False))]

            logger.info("pick next segments {}".format(new_labeled_index))
            labeled_index += new_labeled_index
            logger.info(f"# labeled segments {len(set(labeled_index))}")
            if n_obj == 1:
                labeled_df = df_train.iloc[labeled_index]['o1_gt_anames']
                y_true = labeled_df.apply(lambda anames: gt_udf_name in anames)
            elif n_obj == 2:
                labeled_df = df_train.iloc[labeled_index]['o1_o2_gt_rnames']
                y_true = labeled_df.apply(lambda rnames: gt_udf_name in rnames)
            # log number of positive and negative samples
            logger.info(
                "# positive: {}, # negative: {}".format(
                    sum(y_true), len(y_true) - sum(y_true)
                )
            )
            iter += 1

        logger.info(
            "[{}] # active positive: {}, # active negative: {}, # active total: {}, # random positive: {}, # random negative: {}, # random total: {}".format(
                gt_udf_name,
                pos_count, neg_count, pos_count + neg_count,
                sum(y_true), len(y_true) - sum(y_true), len(y_true)
            )
        )


def extract_udf_candidates(lines, udf_signature_to_gt_udf_name):
    gt_udf_name_to_active_count = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        # Regular expression to match the log pattern
        selection_start_match = re.search(r"UDF selection for (.*?) started", line)
        if selection_start_match:
            udf_signature = selection_start_match.group(1)
        count_match = re.search(r"# positive:\s*(\d+), # negative:\s*(\d+)", line)
        if count_match:
            pos_count = int(count_match.group(1))
            neg_count = int(count_match.group(2))
            gt_udf_name_to_active_count[udf_signature_to_gt_udf_name[udf_signature]] = (pos_count, neg_count)
        i += 1

    return gt_udf_name_to_active_count

async def main():
    # clevrer: python run_udf_selection_random_count.py --num_missing_udfs 3 --query_id 2 --run_id 0 --dataset "clevrer" --query_filename "3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 100 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
    # cityflow: python run_udf_selection_random_count.py --num_missing_udfs 2 --query_id 0 --run_id 0 --dataset "cityflow" --query_filename "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
    # charades: python run_udf_selection_random_count.py --num_missing_udfs 2 --query_id 3 --run_id 0 --dataset "charades" --query_filename "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
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
    parser.add_argument("--ask_for_gt_udf", action="store_true", help="Ask for the gt_udf name interactively if enabled")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--program_with_pixels", action="store_true", help="program with pixels")
    parser.add_argument("--program_with_pretrained_models", action="store_true", help="program with pretrained models")
    parser.add_argument('--generate', action='store_true', help="only run the UDF generation step instead of actually executing the final query.")
    parser.add_argument("--num_workers", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--llm_method", type=str, choices=["gpt", "llava"], default="gpt", help="LLM method for distill model annotations")
    parser.add_argument("--is_async", action="store_true", help="use async for distilled-model UDF labeling")
    parser.add_argument("--openai_model_name", type=str, default="gpt-4-turbo-2024-04-09", help="OpenAI model name")

    args = parser.parse_args()
    num_missing_udfs = args.num_missing_udfs
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    query_filename = args.query_filename
    allow_kwargs_in_udf = args.allow_kwargs_in_udf
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    n_selection_samples = args.n_selection_samples
    ask_for_gt_udf = args.ask_for_gt_udf
    num_interpretations = args.num_interpretations
    program_with_pixels = args.program_with_pixels
    program_with_pretrained_models = args.program_with_pretrained_models
    if program_with_pretrained_models:
        assert program_with_pixels, "program_with_pretrained_models requires program_with_pixels"
    generate = args.generate
    num_workers = args.num_workers
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    llm_method = args.llm_method
    is_async = args.is_async
    openai_model_name = args.openai_model_name

    active_learning_config_name = "ninterp={}-nparams={}-kwargs={}-pixels={}-pretrained_models={}-ntrain_distill={}-nselection_samples={}-selection={}-budget={}-llm_method={}".format(
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
    random_sampling_config_name = f"random_count_{active_learning_config_name}"

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_filename}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]

    output_dir = os.path.join(
        config["output_dir"],
        base_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    base_dir = os.path.join(
        "udf_generation",
        dataset,
        query_filename,
        "num_missing_udfs={}".format(num_missing_udfs),
        random_sampling_config_name,
    )
    log_filename = "qid={}-run={}.log".format(query_id, run_id)
    setup_logging(config, base_dir, log_filename, logger)

    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )
    registered_udfs_json = json.load(open(os.path.join(project_root, "vocaludf", "registered_udfs.json"), "r"))
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

    with open(os.path.join(config['output_dir'], "udf_generation", dataset, query_filename, f"num_missing_udfs={num_missing_udfs}", active_learning_config_name, f"qid={query_id}-run={run_id}.json"), "r") as f:
        generation_output = json.load(f)

    # 1. Retrieve the UDF candidates from the log file
    with open(os.path.join(config['log_dir'], "udf_generation", dataset, query_filename, f"num_missing_udfs={num_missing_udfs}", active_learning_config_name, f"qid={query_id}-run={run_id}.log"), "r") as f:
        lines = f.readlines()
    has_udf_generation = False
    for line in lines:
        if "Shared resources initialization started" in line:
            has_udf_generation = True
            break
    if not has_udf_generation:
        # If the task doesn't contain UDF generation step, simply copy the generation output
        with open(os.path.join(output_dir, "qid={}-run={}.json".format(query_id, run_id)), "w") as f:
            json.dump(generation_output, f)
        return

    # retrieve gt_udf_name, implemented_udfs from the log file
    udf_signature_to_gt_udf_name = {}
    gt_udf_name_to_udf_signature = {}
    for i, line in enumerate(lines):
        if "Proposed functions:" in line:
            # Extract the dictionary from "2024-10-07 04:00:48,823 - vocaludf.async_udf_proposer - INFO - Proposed functions: {'far_from(o0, o1)': 'Whether o0 is far from o1.', 'behind(o0, o1)': 'Whether o0 is behind o1.', 'material_metal(o0)': 'Whether the material of o0 is metal.'}"
            proposed_functions = ast.literal_eval(line.split("Proposed functions: ")[1])
            num_extracted_gt_udf_names = 0
            for j in range(i+1, len(lines)):
                if "Selected gt_udf_name:" in lines[j]:
                    udf_signature = list(proposed_functions.keys())[num_extracted_gt_udf_names]
                    gt_udf_name = lines[j].split("Selected gt_udf_name: ")[1].strip()
                    udf_signature_to_gt_udf_name[udf_signature] = gt_udf_name
                    gt_udf_name_to_udf_signature[gt_udf_name] = udf_signature
                    num_extracted_gt_udf_names += 1
                    if num_extracted_gt_udf_names == len(proposed_functions):
                        break
            i = j + 1
            break
        i += 1
    gt_udf_name_to_active_count = extract_udf_candidates(lines[i:], udf_signature_to_gt_udf_name)
    logger.info("gt_udf_name_to_udf_signature: {}".format(gt_udf_name_to_udf_signature))
    logger.info("gt_udf_name_to_active_count: {}".format(gt_udf_name_to_active_count))

    # 2. Initialize the shared resources
    logger.info("Shared resources initialization started")
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
    )
    logger.info("Shared resources initialization finished")

    # 3. Start the UDF selection process with random sampling
    logger.info("UDF selection started")
    for gt_udf_name, (pos_count, neg_count) in gt_udf_name_to_active_count.items():
        udf_signature = gt_udf_name_to_udf_signature[gt_udf_name]
        udf_name, udf_vars = parse_signature(udf_signature)
        n_obj = len(udf_vars)
        logger.info(f"UDF selection for {udf_signature} started")
        us = RandomUDFSelector(shared_resources, llm_positive_df=None, llm_negative_df=None)
        us.random_count(gt_udf_name, n_obj, pos_count, neg_count)
        logger.info(f"UDF selection for {udf_signature} finished")
    logger.info("UDF selection finished")

    logger.info("Peak memory usuage (in GB): {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0))

if __name__ == "__main__":
    asyncio.run(main())