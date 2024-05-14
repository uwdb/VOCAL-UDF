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
import duckdb
import sys
import resource

# logging.basicConfig()
logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def exception_hook(exc_type, exc_value, exc_traceback, logger=logger):
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )

if __name__ == "__main__":
    # python main.py --query_id 3 --run_id 0 --dataset "clevrer" --budget 20 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 10 --program_with_pixels --cpus 4 --save_labeled_data --n_train_distill 100 --selection_strategy "llm" --selection_labels "user"
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
    parser.add_argument("--program_with_pixels", action="store_true", help="program with pixels")
    parser.add_argument("--program_with_pretrained_models", action="store_true", help="program with pretrained models")
    parser.add_argument("--save_generated_udf", action="store_true", help="save generated UDF to file")
    parser.add_argument("--cpus", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--selection_labels", type=str, choices=["none", "user", "llm"], default="user", help="strategy for UDF selection")

    args = parser.parse_args()
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    allow_kwargs_in_udf = args.allow_kwargs_in_udf
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    ask_for_gt_udf = args.ask_for_gt_udf
    num_interpretations = args.num_interpretations
    program_with_pixels = args.program_with_pixels
    program_with_pretrained_models = args.program_with_pretrained_models
    if program_with_pretrained_models:
        assert program_with_pixels, "program_with_pretrained_models requires program_with_pixels"
    save_generated_udf = args.save_generated_udf
    num_workers = args.cpus
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    selection_labels = args.selection_labels
    if selection_strategy == "both":
        assert selection_labels != "none"
    elif selection_strategy == "model":
        assert selection_labels == "none"

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

    save_udf_base_dir = os.path.join(
        config["output_dir"],
        "udf_generation",
        dataset,
        config_name,
        f"qid={query_id}",
        f"run={run_id}",
    )
    os.makedirs(save_udf_base_dir, exist_ok=True)

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
    # add "BEHIND", "RIGHTOF"
    # registered_functions.extend(
    #     [{
    #         "signature": "right_of(o0, o1)",
    #         "description": "Whether o0 is on the right of o1.",
    #         "function_implementation": "def right_of(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):\n    cx_o1 = (o1_x1 + o1_x2) / 2\n    cx_o2 = (o2_x1 + o2_x2) / 2\n    return cx_o1 > cx_o2"
    #     },
    #     {
    #         "signature": "behind(o0, o1)",
    #         "description": "Whether o0 is behind o1.",
    #         "function_implementation": "def behind(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):\n    center_y_o1 = (o1_y1 + o1_y2) / 2\n    center_y_o2 = (o2_y1 + o2_y2) / 2\n    return center_y_o1 < center_y_o2"
    #     }]
    # )
    available_udf_names = [parse_signature(func["signature"])[0] for func in registered_functions]
    materialized_df_names = []
    on_the_fly_udf_names = []

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
            program_with_pixels,
            program_with_pretrained_models,
            query_id,
            run_id,
            num_workers,
            save_generated_udf,
            save_labeled_data,
            load_labeled_data,
            n_train_distill,
            selection_strategy,
            selection_labels,
            allow_kwargs_in_udf,
            save_udf_base_dir,
        )
        object_domain, relationship_domain, attribute_domain = up.get_active_domain()
        proposed_functions = up.propose(user_query)
        for udf_signature, udf_description in proposed_functions.items():
            # Step 2.a: generate semantic interpretations and implementations. Save the generated UDFs to disk
            # Step 2.b: Distilled-model UDFs
            # udf_candidate_list = up.implement(udf_signature, udf_description)
            udf_candidate_list = up.implement(udf_signature, udf_description)
            # TODO: remove **kwargs from the final implementation
            # TODO: UDF with pixels needs to be materialized before being used in the query

            # # Step 3: Select the best UDF (determine the best UDF between model-based and program-based)
            # NOTE: If we use GPT-4 to provide feedback with zero user effort, how to incorporate the feedback into the UDF selection process?
            # First, retrieve the ground truth UDF
            if ask_for_gt_udf:
                # Ask the user for gt_udf name
                gt_udf_name = input(
                    'Please enter gt_udf_name (options: "near", "far", "right_of", "behind", "location_right", "location_bottom", "color_brown", "color_purple", "color_cyan", "color_yellow", "shape_cylinder", "material_metal"): '
                )
            else:
                # HACK: Use a LM to automatically resolve the ground truth UDF
                # NOTE: Correctness is not guaranteed
                udf_name, udf_vars = parse_signature(udf_signature)
                if len(udf_vars) == 2:
                    gt_udf_candidates = ["near", "far", "right_of", "behind"]
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
                logger.info(f"Selected gt_udf_name: {gt_udf_name}")
            # TODO: since we already precomputed the UDF results, we can directly retrieve them from the database
            selected_udf_candidate = up.select(gt_udf_name, udf_candidate_list)

            semantic_interpretation = selected_udf_candidate.semantic_interpretation
            function_implementation = selected_udf_candidate.function_implementation

            # Assume now that the best UDF is the first one
            # best_impl = implemented_udfs[0]
            logger.info(
                "Best: {}, implementation: {}".format(
                    udf_signature, selected_udf_candidate.function_implementation
                )
            )
            # logger.info(
            #     "Best: {}, implementation: {}".format(
            #         udf_signature, "distilled model"
            #     )
            # )
            # Step 5: Register the UDF
            new_udf = {
                "signature": udf_signature,
                "description": udf_description,
                "semantic_interpretation": semantic_interpretation,  # New field. Unsure if we need this
                "function_implementation": function_implementation,
            }
            registered_functions.append(new_udf)
            if semantic_interpretation == "model":
                materialized_df_names.append(parse_signature(udf_signature)[0])
            else:
                on_the_fly_udf_names.append(parse_signature(udf_signature)[0])
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
        qe = QueryExecutor(config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_df_names, on_the_fly_udf_names, program_with_pixels, num_workers)
        qe.run(parsed_program, y_true, debug=False)
    except Exception as e:
        logger.exception("QueryExecutor Error: {}".format(e))
        logger.info("F1 score: 0")
    logger.info("Peak memory usuage (in KB): {}".forma(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))