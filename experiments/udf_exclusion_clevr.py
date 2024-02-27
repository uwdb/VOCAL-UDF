import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
from functools import partial
import autogen
from visprog.engine.utils import ProgramGenerator, ProgramInterpreter
from visprog.prompts.clevr import create_prompt
import argparse
import logging
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import yaml
import random
from vocaludf.utils import duckdb_execute_cache_sequence, replace_slot
from src.utils import program_to_dsl, dsl_to_program
from vocaludf.main import QueryParser, QueryExecutor
from vocaludf.parser import parse
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    # python udf_exclusion_clevr.py --save_output --output_dir "/gscratch/balazinska/enhaoz/VOCAL-UDF/outputs" --run_id 0 --question_id 0 --method "nl_udf_excluded" --udf "Behind" --llm_model "gpt-4-1106-preview"
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_output', action='store_true', help='save the output')
    parser.add_argument('--output_dir', type=str, help='where to save the results')
    parser.add_argument('--run_id', type=int, help='run id')
    parser.add_argument('--question_id', type=int, help='question id')
    parser.add_argument('--method', type=str, help='method', choices=['nl_udf_included', 'nl_udf_excluded', 'dsl_udf_excluded'])
    parser.add_argument('--udf', type=str, help='UDF name that we evaluate against')
    parser.add_argument('--llm_model', type=str, default="gpt-4-1106-preview", help='llm model', choices=['gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'])
    args = parser.parse_args()
    save_output = args.save_output
    output_dir = args.output_dir
    run_id = args.run_id
    question_id = args.question_id
    method = args.method
    udf = args.udf
    llm_model = args.llm_model

    dataset = "clevr"
    inputs_table_name = "Obj_clevr"

    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))
    prompt_config = yaml.load(open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"), Loader=yaml.FullLoader)
    # Load LLM inference endpoints from an env variable or a file
    # See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
    # and OAI_CONFIG_LIST_sample
    config_list = autogen.config_list_from_json(
        env_or_file="/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        },
    )

    # read json file
    # Fix the test queries, only vary the number of available UDFs
    with open(os.path.join(config['data_dir'], dataset, f"queries_with_{udf}_labels.json"), "r") as f:
        data = json.load(f)
    question = data['questions'][question_id]['question']
    gt_positive_images = data['questions'][question_id]['positive_images']
    gt_dsl = data['questions'][question_id]['dsl']
    logger.info(f"question: {question}")
    logger.info(f"gt_dsl: {gt_dsl}")
    gt_labels = [1 if i in gt_positive_images else 0 for i in range(config[dataset]['dataset_size'])]

    registered_functions = json.load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r"))[dataset]

    if method in ["nl_udf_included", # Use LLM to generate queries from natural language, using all UDFs
                  "nl_udf_excluded", # Use LLM to generate queries from natural language, but with the UDF removed from the available UDFs
                  ]:
        if method == "nl_udf_excluded":
            registered_functions = [func for func in registered_functions if func['signature'].split('(')[0].lower() != udf.lower()]
        qp = QueryParser(config, prompt_config, config_list, dataset, registered_functions, allow_new_udfs=False)
        qp.parse(question)
        parsed_program = qp.get_parsed_program()
        parsed_dsl = qp.get_parsed_query()
        logger.info(f"parsed_program: {parsed_program}")
        logger.info(f"parsed_dsl: {parsed_dsl}")
    elif method == "dsl_udf_excluded":
        # Execute DSL directly but with the UDF removed
        registered_functions = [func for func in registered_functions if func['signature'].split('(')[0].lower() != udf.lower()]
        parsed_program = parse().parseString(gt_dsl, parseAll=True).as_dict()
        # Remove the UDF from the parsed program
        for sg in parsed_program['query']:
            for pred in sg['scene_graph']:
                if udf.lower() == pred['predicate'].lower():
                    sg['scene_graph'].remove(pred)
        logger.info(f"parsed_program: {parsed_program}")
        parsed_dsl = program_to_dsl(parsed_program['query'], rewrite_variables=True, sort_variables=False)
        logger.info(f"parsed_dsl: {parsed_dsl}")
    else:
        raise ValueError("Invalid method")
    qe = QueryExecutor(config, inputs_table_name, registered_functions)
    pred_labels = qe.run(parsed_program, gt_labels, debug=False)

    # Compute accuracy, F1, precision, recall
    accuracy = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"F1: {f1}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    if save_output:
        # Save the output
        output = {
            "question": question,
            "gt_dsl": gt_dsl,
            "dsl": parsed_dsl,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        # create output directory if not exists
        if not os.path.exists(os.path.join(output_dir, 'udf_exclusion', dataset, method, llm_model)):
            os.makedirs(os.path.join(output_dir, 'udf_exclusion', dataset, method, llm_model))

        with open(os.path.join(output_dir, 'udf_exclusion', dataset, method, llm_model, f"udf-{udf}_run-{run_id}_question-{question_id}.json"), "w") as f:
            json.dump(output, f)



