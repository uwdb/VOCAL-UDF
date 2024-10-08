from collections import defaultdict
import time
import yaml
import random
import json
import os
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook, get_active_domain
import logging
import argparse
from sentence_transformers import SentenceTransformer, util
import numpy as np
from vocaludf.query_parser import QueryParser
from vocaludf.async_udf_proposer import SharedResources, UDFProposer, UDFGenerator, UDFSelector
from vocaludf.query_executor import QueryExecutor
import duckdb
import sys
import resource
import asyncio

# logging.basicConfig()
logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s GB'''%(point, usage/1024.0/1024.0 )

async def process_udf(udf_signature, udf_description, shared_resources, gt_udf_name):
    logger.info(f"UDF generation for {udf_signature} started")
    ug = UDFGenerator(shared_resources, udf_signature, udf_description, gt_udf_name)
    udf_candidate_list, llm_positive_df, llm_negative_df = await ug.implement()
    logger.info(f"UDF generation for {udf_signature} finished")

    cost_estimation = ug.get_cost_estimation()
    execution_time = ug.get_execution_time()
    return (
        udf_signature,
        udf_description,
        gt_udf_name,
        udf_candidate_list,
        llm_positive_df,
        llm_negative_df,
        cost_estimation,
        execution_time,
    )


async def main():
    # clevrer: python async_main.py --num_missing_udfs 3 --query_id 2 --run_id 0 --dataset "clevrer" --query_class_name "3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5  --generate --cpus 8 --save_labeled_data --n_train_distill 100 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
    # cityflow: python async_main.py --num_missing_udfs 2 --query_id 0 --run_id 0 --dataset "cityflow" --query_class_name "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --cpus 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
    # charades: python async_main.py --num_missing_udfs 2 --query_id 3 --run_id 0 --dataset "charades" --query_class_name "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --cpus 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"
    # gqa: python main.py --num_missing_udfs 1 --query_id 0 --run_id 0 --dataset "gqa" --query_class_name "unavailable=2-npred=1-nattr_pred=1-nobj_pred=0-nvars=2-min_npos=100-max_npos=5000" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5 --program_with_pixels --generate --cpus 8 --save_labeled_data --n_train_distill 100 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v"
    # vaw: python main.py --num_missing_udfs 1 --query_id 6 --run_id 0 --dataset "vaw" --query_class_name "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5 --program_with_pixels --generate --cpus 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v"
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
    parser.add_argument("--ask_for_gt_udf", action="store_true", help="Ask for the gt_udf name interactively if enabled")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--program_with_pixels", action="store_true", help="program with pixels")
    parser.add_argument("--program_with_pretrained_models", action="store_true", help="program with pretrained models")
    parser.add_argument('--generate', action='store_true', help="only run the UDF generation step instead of actually executing the final query.")
    parser.add_argument("--cpus", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--selection_labels", type=str, choices=["none", "user", "llm"], default="user", help="strategy for UDF selection")
    parser.add_argument("--llm_method", type=str, choices=["gpt4v", "llava"], default="gpt4v", help="LLM method for distill model annotations")
    parser.add_argument("--is_async", action="store_true", help="use async for distilled-model UDF labeling")
    parser.add_argument("--openai_model_name", type=str, default="gpt-4-turbo-2024-04-09", help="OpenAI model name")

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
    ask_for_gt_udf = args.ask_for_gt_udf
    num_interpretations = args.num_interpretations
    program_with_pixels = args.program_with_pixels
    program_with_pretrained_models = args.program_with_pretrained_models
    if program_with_pretrained_models:
        assert program_with_pixels, "program_with_pretrained_models requires program_with_pixels"
    generate = args.generate
    num_workers = args.cpus
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    selection_labels = args.selection_labels
    llm_method = args.llm_method
    is_async = args.is_async
    openai_model_name = args.openai_model_name
    # if selection_strategy != "program":
    #     assert program_with_pixels, "selection_strategy != 'program' requires program_with_pixels"

    if selection_strategy == "both":
        assert selection_labels != "none"
    elif selection_strategy == "model":
        assert selection_labels == "none"

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

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_class_name}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    gt_dsl = input_query["dsl"]
    user_query = input_query["question"]
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
    base_dir = os.path.join(
        "udf_generation",
        dataset,
        query_class_name,
        "num_missing_udfs={}".format(num_missing_udfs),
        config_name,
    )
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        base_dir
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

    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )
    registered_udfs_json = json.load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r"))
    if "single_semantic" in query_class_name:
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

    cost_estimation = defaultdict(float)
    total_execution_time = defaultdict(float)
    udf_generation_execution_time = defaultdict(float)

    # Parse query
    logger.info("Query parsing started")
    _start = time.time()
    qp = QueryParser(
        config, prompt_config, dataset, registered_functions, object_domain, run_id, openai_model_name
    )
    flag = qp.parse(user_query)
    cost_estimation['query_parser'] += qp.get_cost_estimation()
    logger.info("Query parsing finished")
    total_execution_time['query_parsing'] += time.time() - _start
    if 'parse_no' in flag:
        # Step 0: Initialize shared resources
        logger.info("Shared resources initialization started")
        _start = time.time()
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
            query_class_name,
            query_id,
            run_id,
            num_workers,
            save_labeled_data,
            load_labeled_data,
            n_train_distill,
            selection_strategy,
            selection_labels,
            allow_kwargs_in_udf,
            llm_method,
            is_async,
            openai_model_name
        )
        logger.info("Shared resources initialization finished")
        total_execution_time['resource_init'] += time.time() - _start

        # Step 1: propose new UDFs
        logger.info("UDF proposal started")
        _start = time.time()
        up = UDFProposer(shared_resources)
        proposed_functions = up.propose(user_query)
        up_cost_estimation = up.get_cost_estimation()
        for key, value in up_cost_estimation.items():
            cost_estimation[key] += value
        logger.info("UDF proposal finished")
        total_execution_time['udf_proposal'] += time.time() - _start

        # Step 2: generate UDFs, concurrently
        logger.info("UDF generation started")
        _start = time.time()
        task_inputs = []
        for udf_signature, udf_description in proposed_functions.items():
            # First, retrieve the ground truth UDF
            if ask_for_gt_udf:
                # Ask the user for gt_udf name
                if dataset == "clevrer":
                    gt_udf_name = input(
                        'Please enter gt_udf_name (options: "near", "far", "right_of", "behind", "location_right", "location_bottom", "color_brown", "color_purple", "color_cyan", "color_yellow", "shape_cylinder", "material_metal"): '
                    )
                elif dataset == "charades":
                    gt_udf_name = input(
                        'Please enter gt_udf_name (options: "looking_at", "above", "in_front_of", "on_the_side_of", "carrying", "drinking_from", "have_it_on_the_back", "leaning_on", "not_contacting", "standing_on", "twisting", "wiping", "not_looking_at", "beneath", "behind", "in", "covered_by", "eating", "holding", "lying_on", "sitting_on", "touching", "wearing", "writing_on"): '
                    )
                elif dataset == "cityflow":
                    gt_udf_name = input(
                        'Please enter gt_udf_name (options: "suv", "white", "grey", "van", "sedan",  "black",  "red",  "blue", "pickup_truck", "above", "beneath", "to_the_left_of", "to_the_right_of", "in_front_of", "behind"): '
                    )
            else:
                # HACK: Use a LM to automatically resolve the ground truth UDF
                # NOTE: Correctness is not guaranteed
                udf_name, udf_vars = parse_signature(udf_signature)
                if dataset == "clevrer":
                    if len(udf_vars) == 2:
                        gt_udf_candidates = ["near", "far", "right_of", "behind"]
                    else:
                        gt_udf_candidates = ["location_right", "location_bottom", "color_brown", "color_purple", "color_cyan", "color_yellow", "shape_cylinder", "material_metal"]
                elif dataset == "cityflow":
                    if len(udf_vars) == 1:
                        gt_udf_candidates = ["suv", "white", "grey", "van", "sedan", "black", "red", "blue", "pickup_truck"]
                    else:
                        gt_udf_candidates = ["above", "beneath", "to_the_left_of", "to_the_right_of", "in_front_of", "behind"]
                elif dataset == "charades":
                    gt_udf_candidates = ["looking_at", "above", "in_front_of", "on_the_side_of", "carrying", "drinking_from", "have_it_on_the_back", "leaning_on", "not_contacting", "standing_on", "twisting", "wiping", "not_looking_at", "beneath", "behind", "in", "inside", "inside_of", "covered_by", "eating", "holding", "lying_on", "sitting_on", "touching", "wearing", "writing_on"]
                elif dataset == "gqa":
                    if len(udf_vars) == 2:
                        gt_udf_candidates = ["on", "near", "in_front_of", "next_to", "above", "below", "on_top_of", "sitting_on", "carrying", "to_the_left_of", "to_the_right_of", "wearing", "of", "behind", "in", "inside", "inside_of", "by", "on_the_side_of", "holding", "walking_on", "beside"]
                    else:
                        gt_udf_candidates = ["black", "blue", "red", "large", "wood", "tall", "orange", "dark", "pink", "clear", "white", "green", "brown", "gray", "small", "yellow", "metal", "long", "silver", "standing"]
                elif dataset == "vaw":
                    if len(udf_vars) == 2:
                        gt_udf_candidates = ["above", "beneath", "to_the_left_of", "to_the_right_of", "in_front_of", "behind"]
                    else:
                        gt_udf_candidates = ["black", "blue", "brown", "gray", "small", "metal", "long", "dark", "rounded", "orange", "white", "green", "large", "red", "wooden", "yellow", "tall", "silver", "standing", "round"]
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                gt_udf_embeddings = model.encode(gt_udf_candidates)
                implemented_udf_embedding = model.encode([udf_name])
                similarities = util.pytorch_cos_sim(
                    implemented_udf_embedding, gt_udf_embeddings
                )[0]
                gt_udf_name = gt_udf_candidates[similarities.argmax()]
                logger.debug(
                    "similarities: {}".format(
                        [
                            f"{gt_udf_candidate}: {similarity}"
                            for gt_udf_candidate, similarity in zip(
                                gt_udf_candidates, similarities
                            )
                        ]
                    )
                )
                if gt_udf_name in ["inside", "inside_of"]:
                    gt_udf_name = "in"
                logger.info(f"Selected gt_udf_name: {gt_udf_name}")
            task_inputs.append((udf_signature, udf_description, gt_udf_name))

        # Prepare tasks for concurrent execution
        tasks = [
            process_udf(udf_signature, udf_description, shared_resources, gt_udf_name)
            for udf_signature, udf_description, gt_udf_name in task_inputs
        ]

        # Run tasks concurrently
        results = await asyncio.gather(*tasks)
        logger.info("UDF generation finished")
        total_execution_time['udf_generation'] += time.time() - _start

        # Step 3: UDF selection
        # Process results
        logger.info("UDF selection started")
        _start = time.time()
        for result in results:
            (
                udf_signature,
                udf_description,
                gt_udf_name,
                udf_candidate_list,
                llm_positive_df,
                llm_negative_df,
                cost_estimation_per_udf,
                execution_time_per_udf,
            ) = result

            # Update cost estimation from UDF generation
            for key, value in cost_estimation_per_udf.items():
                cost_estimation[key] += value

            # Update execution time from UDF generation
            for key, value in execution_time_per_udf.items():
                udf_generation_execution_time[key] += value

            logger.info(f"UDF selection for {udf_signature} started")
            us = UDFSelector(shared_resources, llm_positive_df, llm_negative_df)
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

            udf_name = parse_signature(udf_signature)[0]
            if semantic_interpretation in ["model", "dummy"]:
                materialized_df_names.append(udf_name)
            else:
                on_the_fly_udf_names.append(udf_name)
        logger.info("UDF selection finished")
        total_execution_time['udf_selection'] += time.time() - _start

        # Step 6: Re-parse the query
        # NOTE: Set allow_new_udfs=False. If the parser still wants to propose new UDFs, we will force it to generate a query that is the best approximation.
        logger.info("Query parsing started")
        _start = time.time()
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
        cost_estimation['query_parser'] += qp.get_cost_estimation()
        logger.info("Query parsing finished")
        total_execution_time['query_parsing'] += time.time() - _start
    if generate:
        output_dir = os.path.join(
            config["output_dir"],
            base_dir
        )
        os.makedirs(output_dir, exist_ok=True)
        parsed_program = qp.get_parsed_program()
        parsed_dsl = qp.get_parsed_query()
        generation_output = dict(
            dataset=dataset,
            object_domain=object_domain,
            relationship_domain=relationship_domain,
            attribute_domain=attribute_domain,
            parsed_program=parsed_program,
            parsed_dsl=parsed_dsl,
            registered_functions=registered_functions,
            available_udf_names=available_udf_names,
            materialized_df_names=materialized_df_names,
            on_the_fly_udf_names=on_the_fly_udf_names,
            program_with_pixels=program_with_pixels,
        )
        logger.info(generation_output)
        # Save the generation output to disk
        with open(os.path.join(output_dir, "qid={}-run={}.json".format(query_id, run_id)), "w") as f:
            json.dump(generation_output, f)
    else: # Deprecated
        try:
            parsed_program = qp.get_parsed_program()
            parsed_dsl = qp.get_parsed_query()
            qe = QueryExecutor(config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_df_names, on_the_fly_udf_names, program_with_pixels, num_workers)
            qe.run(parsed_program, y_true, debug=False)
        except Exception as e:
            logger.exception("QueryExecutor Error: {}".format(e))
            logger.info("F1 score: 0")
    logger.info("Total execution time: {}".format(total_execution_time))
    logger.info("UDF generation execution time: {}".format(udf_generation_execution_time))
    logger.info("Cost estimation: {}".format(cost_estimation))
    logger.info("Peak memory usuage (in GB): {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0))

if __name__ == "__main__":
    asyncio.run(main())