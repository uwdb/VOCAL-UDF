import yaml
import random
import json
import os
from vocaludf.utils import parse_signature, setup_logging, get_active_domain, transform_function
import logging
import argparse
import numpy as np
import duckdb
import sys
import resource
import asyncio
import ast
import re

# logging.basicConfig()
logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

project_root = os.getenv("PROJECT_ROOT")

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s GB'''%(point, usage/1024.0/1024.0 )

# Function to parse kwargs from the id
def parse_kwargs_from_id(udf_id):
    # If there are no kwargs, simply return an empty dict
    if '_' not in udf_id:
        return {}
    base_id, kwargs_parts = udf_id.split('_', 1)

    kwargs = {}
    parts = kwargs_parts.split("_")
    key_parts = []

    for part in parts:
        if '-' in part and part[part.rfind("-")+1:].replace(".", "", 1).lstrip("-").isdigit():
            # Last '-' separates key from the float value
            split_index = part.rfind("-")
            key = "_".join(key_parts + [part[:split_index]])
            value = float(part[split_index + 1:])
            kwargs[key] = value
            key_parts = []  # Reset key_parts for the next key-value pair
        else:
            # Part of the key
            key_parts.append(part)

    return kwargs

def extract_udf_scores(lines):
    # Regular expressions to capture the different log lines
    compute_f1_re = re.compile(r"test F1 score")
    udf_candidate_re = re.compile(r"udf_candidate: (.*?), score: ([0-9.]+)")
    predicted_pos_neg_re = re.compile(r"predicted positive: (\d+), predicted negative: (\d+)")
    positive_neg_re = re.compile(r"positive: (\d+), negative: (\d+)")
    udf_impl_start_re = re.compile(r"UDFCandidate\(id: (.*?), function_implementation: (def .+)")
    impl_and_metadata_re = re.compile(r"(def .+?)\s*,\s*test_score: ([0-9.]+), score: ([0-9.]+), loss_t: \d+", re.DOTALL)

    # Initialize the list for results
    results = []

    i = 0
    while i < len(lines):
        # Step 1: Identify block start and end
        if compute_f1_re.search(lines[i]):
            block_start = i
            while i < len(lines) and not "[Selected]" in lines[i]:
                i += 1
            block_end = i

            # Initialize lists for this block's UDFs
            testing_udfs = {}
            training_udfs = {}

            # Step 2: Parse the block to categorize each UDF
            j = block_start
            while j < block_end:
                match_candidate = udf_candidate_re.search(lines[j])
                if match_candidate:
                    udf_id, score = match_candidate.groups()
                    udf_id, score = udf_id.strip(), float(score)

                    kwargs = parse_kwargs_from_id(udf_id)

                    match_pred_pos_neg = predicted_pos_neg_re.search(lines[j + 1])
                    match_pos_neg = positive_neg_re.search(lines[j + 2])

                    # Check for the function implementation for the first type
                    match_impl_start = udf_impl_start_re.search(lines[j + 3])
                    udf_function_impl = ""
                    if match_impl_start:
                        # Multi-line implementation, capture until "test_score" appears, then split
                        udf_function_impl = match_impl_start.group(2) + "\n"  # keep original indentation
                        j += 4
                        while j < block_end and "test_score" not in lines[j]:
                            udf_function_impl += lines[j]
                            j += 1

                        # Process the final line with test_score and potentially more code
                        final_line = lines[j]
                        # extract the "score" from the final line
                        # Example:    return o1_x1 > o0_x2, test_score: 0.8279289940828403, score: 1.0, loss_t: 0)
                        score = float(final_line.split(", score: ")[1].split(",")[0])
                        match_impl_final = impl_and_metadata_re.search(udf_function_impl + final_line)
                        if match_impl_final:
                            udf_function_impl = match_impl_final.group(1)  # Only the function implementation code

                        testing_udfs[udf_id] = {"function_implementation": udf_function_impl, "kwargs": kwargs, "score": score}
                    else:
                        training_udfs[udf_id] = score
                        j += 3  # Skip the lines read for this UDF
                else:
                    if "No positive samples are labeled. Returning the dummy UDF." in lines[j]:
                        for udf_id, udf_obj in testing_udfs.items():
                            training_udfs[udf_id] = udf_obj["score"]
                        break
                    j += 1

            # Step 3: Remove 'dummy' and find the UDF with the highest score
            combined_udfs = []
            for udf_id, udf_obj in testing_udfs.items():
                if udf_id == "dummy":
                    continue
                score = training_udfs[udf_id]
                combined_udfs.append((udf_id, udf_obj["function_implementation"], udf_obj["kwargs"], score))

            logger.info(f"combined_udfs: {combined_udfs}")

            # Get UDF with highest score (last one in case of a tie)
            highest_score = max([score for _, _, _, score in combined_udfs])
            for udf_id, udf_impl, udf_kwargs, score in combined_udfs:
                if score == highest_score:
                    if udf_id != "model":
                        # Transforms the function by removing **kwargs from the function signature and replacing the kwargs with the actual values
                        udf_impl = transform_function(
                            original_code=udf_impl,
                            instantiation_dict=udf_kwargs,
                        )
                    best_udf = (udf_id, udf_impl) if udf_id == "model" else ("", udf_impl)
            results.append(best_udf)

        i += 1

    for _, best_impl in results:
        logger.info(f"best_impl: {best_impl}")

    return results

async def main():
    # clevrer: python run_udf_selection_no_dummy.py --num_missing_udfs 3 --query_id 2 --run_id 0 --dataset "clevrer" --query_filename "3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 100 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
    # cityflow: python run_udf_selection_no_dummy.py --num_missing_udfs 2 --query_id 0 --run_id 0 --dataset "cityflow" --query_filename "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
    # charades: python run_udf_selection_no_dummy.py --num_missing_udfs 2 --query_id 3 --run_id 0 --dataset "charades" --query_filename "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
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
    # if selection_strategy != "program":
    #     assert program_with_pixels, "selection_strategy != 'program' requires program_with_pixels"


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
    ablation_config_name = f"no_dummy_{active_learning_config_name}"

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_filename}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]

    # Set up logging
    base_dir = os.path.join(
        "udf_generation",
        dataset,
        query_filename,
        "num_missing_udfs={}".format(num_missing_udfs),
        ablation_config_name,
    )
    log_filename = "qid={}-run={}.log".format(query_id, run_id)
    setup_logging(config, base_dir, log_filename, logger)

    output_dir = os.path.join(
        config["output_dir"],
        base_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    registered_udfs_json = json.load(open(os.path.join(project_root, "vocaludf", "registered_udfs.json"), "r"))
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
    gt_udf_name_to_udf_signature = {}
    for i, line in enumerate(lines):
        if "Proposed functions:" in line:
            proposed_functions = ast.literal_eval(line.split("Proposed functions: ")[1])
            num_extracted_gt_udf_names = 0
            for j in range(i+1, len(lines)):
                if "Selected gt_udf_name:" in lines[j]:
                    udf_signature = list(proposed_functions.keys())[num_extracted_gt_udf_names]
                    gt_udf_name = lines[j].split("Selected gt_udf_name: ")[1].strip()
                    gt_udf_name_to_udf_signature[gt_udf_name] = udf_signature
                    num_extracted_gt_udf_names += 1
                    if num_extracted_gt_udf_names == len(proposed_functions):
                        break
            i = j + 1
            break
        i += 1

    logger.info("gt_udf_name_to_udf_signature: {}".format(gt_udf_name_to_udf_signature))

    # 2. Extract the UDF scores
    best_udfs = extract_udf_scores(lines)

    # 3. Start the UDF selection process with random sampling
    logger.info("UDF selection started")
    for i, (gt_udf_name, udf_signature) in enumerate(gt_udf_name_to_udf_signature.items()):
        udf_description = proposed_functions[udf_signature]
        udf_name, udf_vars = parse_signature(udf_signature)

        semantic_interpretation, function_implementation = best_udfs[i]

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