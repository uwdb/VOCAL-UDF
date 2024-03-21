import autogen
import yaml
import random
import json
import os
from vocaludf.utils import parse_signature
import logging
import argparse
from sentence_transformers import SentenceTransformer, util
import numpy as np
from vocaludf.query_parser import QueryParser
from vocaludf.udf_proposer import UDFProposer
from vocaludf.query_executor import QueryExecutor

logging.basicConfig()
logger = logging.getLogger("vocal_udf")
logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    # python main.py --query_id 0 --run_id 0 --dataset "clevrer" --budget 10 --num_interpretations 10
    # python main.py --query_id 0 --run_id 0 --dataset "clevrer" --budget 10 --num_interpretations 10 --allow_kwargs_in_udf --num_parameter_search 10
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in UDF")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument("--ask_for_gt_udf", action="store_true", help="Ask for the gt_udf name interactively if enabled")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")

    args = parser.parse_args()
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    allow_kwargs_in_udf = args.allow_kwargs_in_udf
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    ask_for_gt_udf = args.ask_for_gt_udf
    num_interpretations = args.num_interpretations

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = config[dataset]["input_query_file"]
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    gt_dsl = input_query["dsl"]
    user_query = input_query["question"]
    positive_videos = input_query["positive_videos"]
    y_true = [1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"])]
    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        "udf_generation",
        dataset,
        (
            "budget-{}_ninterp-{}_nparams-{}_with_kwargs".format(
                labeling_budget, num_interpretations, num_parameter_search
            )
            if allow_kwargs_in_udf
            else "budget-{}_ninterp-{}_without_kwargs".format(
                labeling_budget, num_interpretations
            )
        )
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(os.path.join(log_dir, "qid-{}_run-{}.log".format(query_id, run_id)), mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Load LLM inference endpoints from an env variable or a file
    # See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
    # and OAI_CONFIG_LIST_sample
    config_list = autogen.config_list_from_json(
        env_or_file="/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        },
    )

    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )

    registered_functions = json.load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r")
    )["test"]

    # Parse query
    qp = QueryParser(
        config, prompt_config, config_list, dataset, registered_functions, run_id
    )
    flag = qp.parse(user_query)
    if 'no' in flag:
        # Step 1: propose new UDFs
        up = UDFProposer(
            config,
            prompt_config,
            config_list,
            registered_functions,
            dataset,
            labeling_budget,
            num_interpretations,
            num_parameter_search,
            query_id,
            run_id,
            allow_kwargs_in_udf,
        )
        proposed_functions = up.propose(user_query)
        for udf_signature, udf_description in proposed_functions.items():
            # Step 2: generate semantic interpretations and implementations. Save the generated UDFs to disk
            up.implement(udf_signature, udf_description)
            # Step 3: Select the best UDF
            # NOTE: If we use GPT-4 to provide feedback with zero user effort, how to incorporate the feedback into the UDF selection process?
            # First, retrieve the ground truth UDF
            if ask_for_gt_udf:
                # Ask the user for gt_udf name
                gt_udf_name = input(
                    'Please enter gt_udf_name (options: "gt_near.gt_x", "gt_far.gt_x", "gt_rightof.gt_x", "gt_behind.gt_x", "gt_location_right.gt_x", "gt_location_bottom.gt_x", "gt_color_brown.gt_x", "gt_color_purple.gt_x", "gt_color_cyan.gt_x", "gt_color_yellow.gt_x", "gt_shape_cylinder.gt_x", "gt_material_metal.gt_x", where x is a non-negative integer): '
                )
            else:
                # HACK: Use a LM to automatically resolve the ground truth UDF
                # NOTE: Correctness is not guaranteed
                udf_name, udf_vars = parse_signature(udf_signature)
                if len(udf_vars) == 2:
                    gt_udf_candidates = ["near", "far", "rightof", "behind"]
                else:
                    gt_udf_candidates = [
                        "location_right",
                        "location_bottom",
                        "color_brown",
                        "color_purple",
                        "color_cyan",
                        "color_yellow",
                        "shape_cylinder",
                        "material_metal",
                    ]
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                gt_udf_embeddings = model.encode(gt_udf_candidates)
                implemented_udf_embedding = model.encode([udf_name])
                similarities = util.pytorch_cos_sim(
                    implemented_udf_embedding, gt_udf_embeddings
                )[0]
                gt_udf_name = "gt_{}.gt_0".format(gt_udf_candidates[similarities.argmax()])
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
                logger.info(f"Selected gt_udf_name: {gt_udf_name}")
            selected_udf_candidate = up.select(udf_signature, udf_description, gt_udf_name)
            # Assume now that the best UDF is the first one
            # best_impl = implemented_udfs[0]
            logger.info(
                "Best: {}, implementation: {}".format(
                    udf_signature, selected_udf_candidate.function_implementation
                )
            )
            # Step 5: Register the UDF
            new_udf = {
                "signature": udf_signature,
                "description": udf_description,
                "semantic_interpretation": selected_udf_candidate.semantic_interpretation,  # New field. Unsure if we need this
                "python_function": selected_udf_candidate.function_implementation,
            }
            registered_functions.append(new_udf)
        # Step 6: Re-parse the query
        # NOTE: Set allow_new_udfs=False. If the parser still wants to propose new UDFs, we will force it to generate a query that is the best approximation.
        qp = QueryParser(
            config,
            prompt_config,
            config_list,
            dataset,
            registered_functions,
            run_id,
            allow_new_udfs=False,
        )
        qp.parse(user_query)
    try:
        parsed_program = qp.get_parsed_program()
        parsed_dsl = qp.get_parsed_query()
        qe = QueryExecutor(config, f"Obj_{dataset}", registered_functions)
        qe.run(parsed_program, y_true, debug=False)
        # TODO: Only register UDFs that are actually used in the query and when the F1 score is above a certain threshold
    except Exception as e:
        logger.error("QueryExecutor Error: {}".format(e))
        logger.info("F1 score: 0")
