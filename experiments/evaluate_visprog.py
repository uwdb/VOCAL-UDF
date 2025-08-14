import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
from functools import partial
import duckdb
from visprog.engine.utils import ProgramGenerator, ProgramInterpreter
import argparse
import logging
from tqdm import tqdm
import json
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import yaml
import random

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

project_root = os.getenv("PROJECT_ROOT")

if __name__ == "__main__":
    # VAW: python evaluate_visprog.py --use_precomputed --num_missing_udfs 1 --dataset "vaw" --query_filename "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000" --run_id 0 --query_id 0 --llm_model "gpt-4-turbo-2024-04-09"
    # CLEVRER: python evaluate_visprog.py --use_precomputed --num_missing_udfs 1 --dataset "clevrer" --query_filename "3_new_udfs_labels" --run_id 0 --query_id 0 --llm_model "gpt-4-turbo-2024-04-09"
    # Charades: python evaluate_visprog.py --use_precomputed --num_missing_udfs 0 --dataset "charades" --query_filename "unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2" --run_id 0 --query_id 0 --llm_model "gpt-4-turbo-2024-04-09"
    # CityFlow: python evaluate_visprog.py --use_precomputed --num_missing_udfs 0 --dataset "cityflow" --query_filename "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" --run_id 0 --query_id 0 --llm_model "gpt-4-turbo-2024-04-09"
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_precomputed', action='store_true', help='use precomputed object detection results')
    parser.add_argument("--num_missing_udfs", type=int, help="number of missing UDFs")
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument("--query_filename", type=str, help="query filename")
    parser.add_argument('--run_id', type=int, help='run id')
    parser.add_argument('--query_id', type=int, help='query id')
    parser.add_argument('--llm_model', type=str, default="gpt-4-turbo-2024-04-09", help='llm model', choices=['gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-1106', 'gpt-4-turbo-2024-04-09', "gpt-4o"])
    args = parser.parse_args()
    use_precomputed = args.use_precomputed
    assert use_precomputed, "use_precomputed must be True"
    num_missing_udfs = args.num_missing_udfs
    dataset = args.dataset
    query_filename = args.query_filename
    run_id = args.run_id
    query_id = args.query_id
    llm_model = args.llm_model

    random.seed(run_id)

    config = yaml.safe_load(open(os.path.join(project_root, "configs", "config.yaml"), "r"))

    input_vids = list(range(config[dataset]["dataset_size"] // 2, config[dataset]["dataset_size"]))

    # read json file
    # Fix the test queries, only vary the number of available UDFs
    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_filename}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    gt_dsl = input_query["dsl"]
    user_query = input_query["question"]
    positive_videos = input_query["positive_videos"]
    logger.debug(f"positive_videos: {positive_videos}")
    new_modules = input_query["new_modules"]
    y_true = [1 if i in positive_videos else 0 for i in input_vids]
    logger.debug(f"y_true: {y_true}")

    """
    Set up logging
    """
    base_dir = os.path.join(
        "query_execution",
        dataset,
        query_filename,
        "num_missing_udfs={}".format(num_missing_udfs),
        f"visprog-llm={llm_model}",
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

    # Base modules
    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )
    registered_udfs_json = json.load(open(os.path.join(project_root, "vocaludf", "registered_udfs.json"), "r"))
    if dataset == "clevrer":
        module_list = ["LOC", "TRACK", "GRAY", "RED", "BLUE", "GREEN", "CUBE", "SPHERE", "RUBBER", "LEFTOF", "FRONTOF", "LEFT", "TOP", "EVENT", "BEFORE", "EVAL", "RESULT"]
    elif dataset == "cityflow":
        module_list = ["LOC", "TRACK", "EVENT", "BEFORE", "EVAL", "RESULT"]
        module_list.extend([parse_signature(func["signature"])[0].replace("_", "").upper() for func in registered_udfs_json[f"{dataset}_base"]])
    elif dataset == "charades":
        module_list = ["LOC", "TRACK", "EVENT", "BEFORE", "EVAL", "RESULT", "PERSON", "BAG", "BED", "BLANKET", "BOOK", "BOX", "BROOM", "CHAIR", "CLOSETCABINET", "CLOTHES", "CUPGLASSBOTTLE", "DISH", "DOOR", "DOORKNOB", "DOORWAY", "FLOOR", "FOOD", "GROCERIES", "LAPTOP", "LIGHT", "MEDICINE", "MIRROR", "PAPERNOTEBOOK", "PHONECAMERA", "PICTURE", "PILLOW", "REFRIGERATOR", "SANDWICH", "SHELF", "SHOE", "SOFACOUCH", "TABLE", "TELEVISION", "TOWEL", "VACUUM", "WINDOW"]
        for func in registered_udfs_json[f"{dataset}_base"]:
            if func["signature"] == "object(o0, name)":
                continue
            else:
                module_list.append(parse_signature(func["signature"])[0].replace("_", "").upper())
    else:
        raise NotImplementedError(f"dataset: {dataset} is not implemented")

    new_modules = input_query["new_modules"]
    assert num_missing_udfs >= 0 and num_missing_udfs <= len(new_modules), "num_missing_udfs must be between 0 and len(new_modules)"
    for new_module in new_modules[:(len(new_modules)-num_missing_udfs)]:
        module_list.append(new_module)

    logger.info(f"module_list: {module_list}")
    interpreter = ProgramInterpreter(dataset=dataset, use_precomputed=use_precomputed, module_list=module_list)

    prompt_modules = '\n'.join(module_list)
    prompt_modules = f"You can only use modules below to generate the program:\n{prompt_modules}\nIf you think the user's request cannot be resolved using the provided modules, try your best to generate a program that best approximates the user's intent."

    if dataset == "charades":
        from visprog.prompts.charades import create_prompt
    elif dataset == "clevrer":
        from visprog.prompts.clevrer import create_prompt
    elif dataset == "cityflow":
        from visprog.prompts.cityflow import create_prompt
    else:
        raise NotImplementedError(f"dataset: {dataset} is not implemented")

    prompter = partial(create_prompt, method='random', num_prompts=8, seed=run_id, prompt_modules=prompt_modules)
    generator = ProgramGenerator(prompter=prompter, temperature=config['visprog']['program_generator']['temperature'],top_p=config['visprog']['program_generator']['top_p'], llm_model=llm_model, run_id=run_id)

    messages = []
    for retry in range(3):
        logger.info(f"retry: {retry}")
        prog,_ = generator.generate(dict(question=user_query),retry, messages)
        logger.info(f"generated prog: {prog}")

        pred_positive_videos = []
        failed = 0
        try:
            for vid in tqdm(input_vids):
                video = []
                if not use_precomputed:
                    raise NotImplementedError("Object detection results are not precomputed")
                init_state = dict(
                    VIDEO=video,
                    vid=vid
                )
                result, prog_state = interpreter.execute(prog,init_state,inspect=False)
                if result == "yes":
                    pred_positive_videos.append(vid)
            logger.info(pred_positive_videos)
            break
        except Exception as e:
            logger.exception(f"""
            Error while executing program: {e}. Please fix it and regenerate a program based on the question, with no other comments, inline comments, syntax highlighter, explanations, reasoning, or dialogue. The generated program should be wrapped by
            ```python
            ```
            """)
            failed += 1
            messages.extend([
                {"role": "assistant", "content": prog},
                {"role": "user", "content": str(e)},
            ])
    logger.info(f"pred_positive_videos: {pred_positive_videos}")
    y_pred = [1 if vid in pred_positive_videos else 0 for vid in input_vids]
    logger.debug(f"y_true: {y_true}")
    logger.debug(f"y_pred: {y_pred}")

    # Compute accuracy, F1, precision, recall
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"F1: {f1}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"# Failures: {failed}")
