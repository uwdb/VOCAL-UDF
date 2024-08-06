import yaml
import random
import json
from vocaludf.utils import StreamToLogger, exception_hook
import logging
import numpy as np
import argparse
import os
import sys

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

def main(query_id, run_id, dataset, query_class_name, allow_kwargs_in_udf, num_parameter_search, labeling_budget, n_selection_samples, llm_method, num_interpretations, program_with_pixels, program_with_pretrained_models, n_train_distill, selection_strategy, selection_labels):
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
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

    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        "best_udf_type",
        dataset,
        query_class_name,
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

    if dataset == "clevrer":
        num_missing_udfs = 3
    elif dataset in ["cityflow", "charades"]:
        num_missing_udfs = 2

    with open(
        os.path.join(
            config["log_dir"],
            "udf_generation",
            dataset,
            query_class_name,
            f"num_missing_udfs={num_missing_udfs}",
            config_name,
            "qid={}-run={}.log".format(query_id, run_id),
        ),
        "r",
    ) as f:
        lines = f.readlines()

    udf_logs = {}
    for i, line in enumerate(lines):
        if "Implementing UDF: " in line:
            udf_name = line.split("Implementing UDF: ")[1].split("(")[0].strip()
        if "compute test F1 score" in line:
            start = i
        if "Best: " in line:
            end = i
            udf_logs[udf_name] = lines[start:end]

    output_dict = {}
    for udf_name, udf_log in udf_logs.items():
        logger.info(f"Processing UDF: {udf_name}")
        candidates = {}
        best_udf_ids = set()
        best_udf_types = set()
        best_test_score = -1
        selected_udf_id = None
        selected_udf_type = None
        selected_test_score = -1
        for i, line in enumerate(udf_log):
            if "test_score" in line:
                test_score = float(line.split("test_score: ")[1].split(",")[0])
                j = 0
                while "UDFCandidate" not in udf_log[i - j]:
                    j += 1
                udf_id = udf_log[i - j].split("UDFCandidate(id: ")[1].split(",")[0]
                if "[Selected]: UDFCandidate" in udf_log[i - j]:
                    selected_test_score = test_score
                    selected_udf_id = udf_id
                    if udf_id == "model":
                        selected_udf_type = "model"
                    elif udf_id == "dummy":
                        selected_udf_type = "dummy"
                    else:
                        selected_udf_type = "program"
                    logger.info(f"Selected UDF ID: {udf_id}, Test Score: {test_score}")
                    output_dict[udf_name] = {
                        "best_udf_ids": list(best_udf_ids),
                        "best_test_score": best_test_score,
                        "best_udf_types": list(best_udf_types),
                        "selected_udf_id": selected_udf_id,
                        "selected_test_score": selected_test_score,
                        "selected_udf_type": selected_udf_type,
                        "candidates": candidates
                    }
                    break
                else:
                    logger.info(f"UDF ID: {udf_id}, Test Score: {test_score}")
                    program_types = []
                    if udf_id == "model":
                        udf_type = "model"
                    elif udf_id == "dummy":
                        udf_type = "dummy"
                    else:
                        udf_type ="program"
                        if len(udf_id.split("_")) > 1:
                            program_types.append("parameter")
                        col_names = ["_oname", "_anames", "_rnames"]
                        for k in range(i - j + 1, i + 1):
                            if any(s in udf_log[k] for s in col_names):
                                program_types.append("reuse")
                                break
                        for k in range(i - j + 1, i + 1):
                            if "img" in udf_log[i - j] and "img" in udf_log[k]:
                                program_types.append("pixel")
                                break
                    candidates[udf_id] = {
                        "udf_type": udf_type,
                        "program_types": program_types,
                        "test_score": test_score
                    }
                    if test_score >= best_test_score:
                        if test_score > best_test_score:
                            best_test_score = test_score
                            best_udf_ids = set()
                            best_udf_types = set()
                        best_udf_ids.add(udf_id)
                        best_udf_types.add(udf_type)

    output_dir = os.path.join(
        config["output_dir"],
        "best_udf_type",
        dataset,
        query_class_name,
        config_name,
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "qid={}-run={}.json".format(query_id, run_id)), "w") as f:
            json.dump(output_dict, f)

if __name__ == '__main__':
    # clevrer: python evaluate_best_udf_type.py --query_id 0 --run_id 0 --dataset "clevrer" --query_class_name "3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5 --n_train_distill 100 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v"
    # charades: python evaluate_best_udf_type.py --query_id 0 --run_id 0 --dataset "charades" --query_class_name "unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2" --budget 50 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --num_parameter_search 5 --n_train_distill 500 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v"
    # cityflow: python evaluate_best_udf_type.py --query_id 0 --run_id 0 --dataset "cityflow" --query_class_name "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" --budget 50 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --num_parameter_search 5 --n_train_distill 500 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v"
    # CLEVRER
    dataset = "clevrer"
    query_class_name = "3_new_udfs_labels"
    for query_id in range(30):
        for run_id in range(3):
            try:
                main(query_id, run_id, dataset, query_class_name, allow_kwargs_in_udf=True, num_parameter_search=5, labeling_budget=20, n_selection_samples=500, llm_method="gpt4v", num_interpretations=10, program_with_pixels=True, program_with_pretrained_models=False, n_train_distill=100, selection_strategy="both", selection_labels="user")
            except Exception as e:
                logger.error(e)


    # Charades
    dataset = "charades"
    query_class_names = [
        "unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2",
        "unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2",
        "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2"
    ]
    for query_class_name in query_class_names:
        for query_id in range(10):
            for run_id in range(3):
                try:
                    main(query_id, run_id, dataset, query_class_name, allow_kwargs_in_udf=True, num_parameter_search=5, labeling_budget=50, n_selection_samples=500, llm_method="gpt4v", num_interpretations=10, program_with_pixels=False, program_with_pretrained_models=False, n_train_distill=500, selection_strategy="both", selection_labels="user")
                except Exception as e:
                    logger.error(e)

    # CityFlow
    dataset = "cityflow"
    query_class_names = [
        "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737",
        "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737"
    ]
    for query_class_name in query_class_names:
        for query_id in range(15):
            for run_id in range(3):
                try:
                    main(query_id, run_id, dataset, query_class_name, allow_kwargs_in_udf=True, num_parameter_search=5, labeling_budget=50, n_selection_samples=500, llm_method="gpt4v", num_interpretations=10, program_with_pixels=False, program_with_pretrained_models=False, n_train_distill=500, selection_strategy="both", selection_labels="user")
                except Exception as e:
                    logger.error(e)