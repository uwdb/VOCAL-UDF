import yaml
import random
import json
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook, get_active_domain
import logging
import numpy as np
import argparse
import os
import sys
from vocaludf.async_udf_generator import UDFGenerator
import ast
from openai import OpenAI, AsyncOpenAI
from collections import defaultdict
import asyncio

client = OpenAI()

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

class LlmDecidesUdfType(UDFGenerator):
    def __init__(
        self,
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
        openai_model_name,
        udf_signature,
        udf_description
    ):
        self.config = config
        self.prompt_config = prompt_config
        self.registered_functions = registered_functions
        self.object_domain = object_domain
        self.relationship_domain = relationship_domain
        self.attribute_domain = attribute_domain
        self.dataset = dataset
        self.labeling_budget = labeling_budget
        self.n_selection_samples = n_selection_samples
        self.num_interpretations = num_interpretations
        self.num_parameter_search = num_parameter_search
        self.program_with_pixels = program_with_pixels
        self.program_with_pretrained_models = program_with_pretrained_models
        self.query_id = query_id
        self.run_id = run_id
        self.num_workers = num_workers
        self.selection_strategy = selection_strategy
        self.allow_kwargs_in_udf = allow_kwargs_in_udf
        self.llm_method = llm_method
        self.is_async = is_async
        self.openai_model_name = openai_model_name

        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.udf_signature = udf_signature
        self.udf_description = udf_description
        self.cost_estimation = defaultdict(float)

async def main():
    # clevrer: python evaluate_llm_decides_udf_type.py --query_id 0 --run_id 0 --dataset "clevrer" --query_filename "3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5 --n_train_distill 100 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4-turbo-2024-04-09"
    # charades: python evaluate_llm_decides_udf_type.py --query_id 0 --run_id 0 --dataset "charades" --query_filename "unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2" --budget 50 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --num_parameter_search 5 --n_train_distill 500 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4-turbo-2024-04-09"
    # cityflow: python evaluate_llm_decides_udf_type.py --query_id 0 --run_id 0 --dataset "cityflow" --query_filename "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" --budget 50 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --num_parameter_search 5 --n_train_distill 500 --selection_strategy "both" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4-turbo-2024-04-09"

    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--query_filename", type=str, help="query filename")
    parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in UDF")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument("--n_selection_samples", type=int, default=500, help="number of sampled videos per selection iteration")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--program_with_pixels", action="store_true", help="program with pixels")
    parser.add_argument("--program_with_pretrained_models", action="store_true", help="program with pretrained models")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--llm_method", type=str, choices=["gpt4v", "llava"], default="gpt4v", help="LLM method for distill model annotations")
    parser.add_argument("--is_async", action="store_true", help="use async for distilled-model UDF labeling")
    parser.add_argument("--openai_model_name", type=str, help="OpenAI model name")

    args = parser.parse_args()
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    query_filename = args.query_filename
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
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    is_async = args.is_async
    openai_model_name = args.openai_model_name

    config_name = "ninterp={}-nparams={}-kwargs={}-pixels={}-pretrained_models={}-ntrain_distill={}-nselection_samples={}-selection={}-budget={}-llm_method={}".format(
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

    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_filename}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]

    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        "llm_decides_udf_type",
        dataset,
        query_filename,
        config_name,
        openai_model_name,
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

    if dataset == "clevrer":
        num_missing_udfs = 3
    elif dataset in ["cityflow", "charades"]:
        num_missing_udfs = 2

    with open(
        os.path.join(
            config["log_dir"],
            "udf_generation",
            dataset,
            query_filename,
            f"num_missing_udfs={num_missing_udfs}",
            config_name,
            "qid={}-run={}.log".format(query_id, run_id),
        ),
        "r",
    ) as f:
        lines = f.readlines()
    proposed_functions = {}
    for line in lines:
        # Proposed functions: {'far_from(o0, o1)': 'Whether o0 is far from o1', 'behind(o0, o1)': 'Whether o0 is behind o1', 'material_metal(o0)': 'Whether the material of o0 is metal'}
        if "INFO - Proposed functions: " in line:
            proposed_functions = line.split("INFO - Proposed functions: ")[1].strip()
            proposed_functions = ast.literal_eval(proposed_functions)
            break

    logger.info("Proposed functions: {}".format(proposed_functions))

    registered_udfs_json = json.load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r"))
    registered_functions = registered_udfs_json[f"{dataset}_base"]
    new_modules = input_query["new_modules"]
    assert num_missing_udfs >= 0 and num_missing_udfs <= len(new_modules), "num_missing_udfs must be between 0 and len(new_modules)"
    for new_module in new_modules[:(len(new_modules)-num_missing_udfs)]:
        registered_functions.append(registered_udfs_json[dataset][new_module])
    logger.info("Registered functions: {}".format(registered_functions))
    object_domain, relationship_domain, attribute_domain = get_active_domain(config, dataset, registered_functions)
    logger.info("Active domains: object={}, relationship={}, attribute={}".format(object_domain, relationship_domain, attribute_domain))

    num_workers = 1
    save_labeled_data = False
    load_labeled_data = False
    output_dict = {}
    for udf_signature, udf_description in proposed_functions.items():
        up = LlmDecidesUdfType(
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
            openai_model_name,
            udf_signature,
            udf_description
        )
        llm_decision = await up._llm_decides_udf_type()
        udf_type = ""
        if llm_decision == "programUDF":
            udf_type = "program"
        elif llm_decision == "modelUDF":
            udf_type = "model"
        else:
            logger.error(f"Unknown LLM decision: {llm_decision}")
        udf_name, udf_vars = parse_signature(udf_signature)
        output_dict[udf_name] = udf_type
        logger.info(f"LLM decides UDF type: {udf_name} -> {udf_type}")

    output_dir = os.path.join(
        config["output_dir"],
        "llm_decides_udf_type",
        dataset,
        query_filename,
        config_name,
        openai_model_name,
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "qid={}-run={}.json".format(query_id, run_id)), "w") as f:
            json.dump(output_dict, f)

if __name__ == '__main__':
    asyncio.run(main())