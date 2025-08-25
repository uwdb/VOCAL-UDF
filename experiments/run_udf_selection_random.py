from collections import defaultdict
import time
import pandas as pd
import yaml
import random
import json
import os
from vocaludf.utils import parse_signature, setup_logging, get_active_domain, transform_function, SharedResources, UDFCandidate
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
    def _select(self, gt_udf_name, udf_candidate_list, df_with_img_column):
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
        df_train, df_test = self.construct_train_and_test_data(n_obj, self.n_train, self.n_test, df_with_img_column=df_with_img_column)

        # Select new video segments to label
        # TODO: df_train and llm_positive_df may contain the same tuples
        labeled_index = []

        segment_selection_time = 0
        _start_segment_selection_time = time.time()

        for iter in range(self.labeling_budget):
            logger.info("iter {}: ".format(iter))
            _start_segment_selection_time_per_iter = time.time()

            unlabeled_index = np.setdiff1d(
                np.arange(len(df_train)), labeled_index, assume_unique=True
            )
            logger.debug("len(unlabeled_index): {}".format(len(unlabeled_index)))

            # Randomly select one
            new_labeled_index = [int(np.random.choice(unlabeled_index, 1, replace=False))]

            logger.info("pick next segments {}".format(new_labeled_index))
            labeled_index += new_labeled_index
            new_labeled_df = df_train.iloc[new_labeled_index]
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

            # Update scores
            indices_to_remove = []
            for i in range(len(udf_candidate_list)):
                labeled_df = df_train.iloc[labeled_index]
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
                labeled_df = df_train.iloc[labeled_index]
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


def extract_udf_candidates(lines, udf_signature_to_gt_udf_name):
    results = {}
    gt_udf_name_to_best_ckpt = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        # Regular expression to match the log pattern
        info_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - .* - INFO - \[(.*?)\] \[(\d+)\] (\w+): (.*)$', line)
        debug_match = re.match(
            r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - .* - DEBUG - \[(.*?)\] Best model checkpoint: (.*)$',
            line
        )
        if info_match:
            timestamp, udf_signature, idx, field_name, value = info_match.groups()
            idx = int(idx)
            # Initialize data structures if necessary
            if udf_signature not in results:
                results[udf_signature] = {}
            if idx not in results[udf_signature]:
                results[udf_signature][idx] = {}
            if field_name == 'function_implementation':
                # Read the function implementation, including subsequent lines
                function_implementation = value + '\n'
                i += 1
                while i < len(lines) and not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+', lines[i]):
                    function_implementation += lines[i] + '\n'
                    i +=1
                results[udf_signature][idx][field_name] = function_implementation.rstrip('\n')
                continue  # Skip incrementing i here because it's already done
            elif field_name == 'kwargs':
                # Read kwargs, possibly spanning multiple lines
                kwargs_str = value
                i += 1
                while i < len(lines) and not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+', lines[i]):
                    kwargs_str += '\n' + lines[i]
                    i +=1
                # Safely evaluate the kwargs string
                try:
                    kwargs = ast.literal_eval(kwargs_str)
                except Exception as e:
                    kwargs = {}
                results[udf_signature][idx][field_name] = kwargs
                continue
            else:
                # For semantic_interpretation and other fields
                results[udf_signature][idx][field_name] = value
        elif debug_match:
            timestamp, udf_signature, best_ckpt = debug_match.groups()
            # Store the best_ckpt for the udf_signature; overwrite if multiple entries
            gt_udf_name_to_best_ckpt[udf_signature_to_gt_udf_name[udf_signature]] = best_ckpt.strip()
        i += 1

    # Convert the results into the desired format: a dictionary of lists
    gt_udf_name_to_implemented_udfs = {}
    for udf_signature, idx_dict in results.items():
        # Convert idx_dict to a list sorted by idx
        idx_items = [idx_dict[idx] for idx in sorted(idx_dict.keys())]
        gt_udf_name_to_implemented_udfs[udf_signature_to_gt_udf_name[udf_signature]] = idx_items

    return gt_udf_name_to_implemented_udfs, gt_udf_name_to_best_ckpt

async def main():
    # clevrer: python run_udf_selection_random.py --num_missing_udfs 3 --query_id 2 --run_id 0 --dataset "clevrer" --query_filename "3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 100 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
    # cityflow: python run_udf_selection_random.py --num_missing_udfs 2 --query_id 0 --run_id 0 --dataset "cityflow" --query_filename "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
    # charades: python run_udf_selection_random.py --num_missing_udfs 2 --query_id 3 --run_id 0 --dataset "charades" --query_filename "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
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
    random_sampling_config_name = f"random_{active_learning_config_name}"

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
    available_udf_names = [parse_signature(func["signature"])[0] for func in registered_functions]
    materialized_df_names = []
    on_the_fly_udf_names = []

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
                elif "filtering out functions that are already registered" in lines[j]:
                    for k in range(j+1, len(lines)):
                        if "filtering out" in lines[k]:
                            filtered_out_udf_signature = lines[k].split("filtering out ")[1].strip()
                            del proposed_functions[filtered_out_udf_signature]
                        else:
                            break
            i = j + 1
            break
        i += 1
    gt_udf_name_to_implemented_udfs, gt_udf_name_to_best_ckpt = extract_udf_candidates(lines[i:], udf_signature_to_gt_udf_name)
    logger.info("gt_udf_name_to_implemented_udfs: {}".format(gt_udf_name_to_implemented_udfs))
    logger.info("gt_udf_name_to_best_ckpt: {}".format(gt_udf_name_to_best_ckpt))

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
    for gt_udf_name, implemented_udfs in gt_udf_name_to_implemented_udfs.items():
        best_ckpt = gt_udf_name_to_best_ckpt[gt_udf_name]
        udf_signature = gt_udf_name_to_udf_signature[gt_udf_name]
        udf_description = proposed_functions[udf_signature]
        udf_name, udf_vars = parse_signature(udf_signature)
        # Prepare the UDF candidate list
        # Read UDF candidates from json files
        logger.info(f"Preparing UDF candidates for {udf_signature} started")
        udf_candidate_list = []  # List[UDFCandidate]
        for i in range(len(implemented_udfs)):
            try:
                udf_dict = implemented_udfs[i]
                udf_dict["udf_name"] = udf_name
                udf_dict["udf_signature"] = udf_signature
                udf_dict["udf_description"] = udf_description
                if allow_kwargs_in_udf and udf_dict.get("kwargs", {}):
                    # Instantiate the kwargs with default values
                    udf_variant_dict = copy.deepcopy(udf_dict)
                    udf_variant_dict["kwargs"] = {k: v["default"] for k, v in udf_variant_dict["kwargs"].items() if v["default"] is not None}
                    new_udf_candidate = UDFCandidate(id=i, payload=udf_variant_dict)
                    logger.debug(f"[{udf_signature}] {new_udf_candidate}")
                    udf_candidate_list.append(new_udf_candidate)
                    # Instantiate the kwargs with values randomly sampled from the range
                    if num_parameter_search and num_parameter_search > 0:
                        for _ in range(num_parameter_search):
                            # deepcopy udf_dict
                            udf_variant_dict = copy.deepcopy(udf_dict)
                            for k in list(udf_variant_dict["kwargs"].keys()):
                                # randomly sample a value from the range
                                udf_variant_dict["kwargs"][k] = np.random.uniform(udf_variant_dict["kwargs"][k]["min"], udf_variant_dict["kwargs"][k]["max"])
                            new_udf_candidate = UDFCandidate(id=i, payload=udf_variant_dict)
                            logger.debug(f"[{udf_signature}] {new_udf_candidate}")
                            udf_candidate_list.append(new_udf_candidate)
                else: # No additional arguments
                    new_udf_candidate = UDFCandidate(id=i, payload=udf_dict)
                    logger.debug(f"[{udf_signature}] {new_udf_candidate}")
                    udf_candidate_list.append(new_udf_candidate)
            except Exception as e:
                logger.exception(f"[{udf_signature}] Failed to read UDF candidate: {e}")
        udf_dict = {}
        udf_dict["udf_name"] = udf_name
        udf_dict["udf_signature"] = udf_signature
        udf_dict["udf_description"] = udf_description
        udf_dict["semantic_interpretation"] = 'model'
        udf_dict["function_implementation"] = best_ckpt
        new_udf_candidate = UDFCandidate(id='model', payload=udf_dict)
        udf_candidate_list.append(new_udf_candidate)
        logger.debug(f"[{udf_signature}] {new_udf_candidate}")
        logger.info(f"Preparing UDF candidates for {udf_signature} finished")

        logger.info(f"UDF selection for {udf_signature} started")
        us = RandomUDFSelector(shared_resources, llm_positive_df=None, llm_negative_df=None)
        selected_udf_candidate = us.select(gt_udf_name, udf_candidate_list)
        logger.info(f"UDF selection for {udf_signature} finished")
        if selected_udf_candidate is None:
            logger.warning("No UDF candidate is selected. Skipping...")
            continue

        semantic_interpretation = selected_udf_candidate.semantic_interpretation
        function_implementation = selected_udf_candidate.function_implementation

        logger.info(
            "Best: {}, implementation: {}".format(
                udf_signature, function_implementation
            )
        )

        # Prepare the new UDF data
        new_udf = {
            "signature": udf_signature,
            "description": udf_description,
            "semantic_interpretation": semantic_interpretation, # New field. Unsure if we need this
            "function_implementation": function_implementation,
        }
        registered_functions.append(new_udf)

        if semantic_interpretation in ["model", "dummy"]:
            materialized_df_names.append(udf_name)
        else:
            on_the_fly_udf_names.append(udf_name)
    logger.info("UDF selection finished")

    if generate:
        generation_output["registered_functions"] = registered_functions
        generation_output["materialized_df_names"] = materialized_df_names
        generation_output["on_the_fly_udf_names"] = on_the_fly_udf_names
        logger.info(generation_output)
        # Save the generation output to disk
        with open(os.path.join(output_dir, "qid={}-run={}.json".format(query_id, run_id)), "w") as f:
            json.dump(generation_output, f)
    else: # Deprecated
        raise NotImplementedError
    logger.info("Peak memory usuage (in GB): {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0))

if __name__ == "__main__":
    asyncio.run(main())