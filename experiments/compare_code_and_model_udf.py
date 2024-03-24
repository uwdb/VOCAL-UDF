# TODO: need to show that for some UDFs, class 3a fails, class 3b is expensive (derive a formula in terms of $$ cost if running over the entire dataset), and class 4 succeeds.
## Try color, shape, materials over clevrer.
## Try more difficult relationships/attributes from other datasets

import argparse
import logging
import os
import random
import yaml
import numpy as np
import cv2
import duckdb
import base64
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import Dataset
import lightning.pytorch as pl
from vocaludf import mlp
from collections import defaultdict
from vocaludf.utils import parse_signature
from vocaludf.udf_proposer import CodeUDFWithPixelsProposer
from vocaludf.model_udf import ModelDistiller
from sentence_transformers import SentenceTransformer, util


client = OpenAI()

logging.basicConfig()
logger = logging.getLogger("vocal_udf")
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    # Code-based UDF
    # python compare_code_and_model_udf.py --run_id 1 --dataset "clevrer" --udf_class "material_metal" --code_based --num_interpretations 10

    # Model-based UDF
    # python compare_code_and_model_udf.py --run_id 0 --dataset "clevrer" --udf_class "color_brown" --model_based --n_train 100 --save_labeled_data --load_labeled_data
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--udf_class", type=str, help="UDF class name we want to generate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--code_based", action="store_true", help="code-based UDF")
    group.add_argument("--model_based", action="store_true", help="model-based UDF")
    parser.add_argument("--n_train", type=int, help="number of training samples")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data")
    # parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in UDF")
    # parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    # parser.add_argument("--budget", type=int, help="labeling budget")
    # parser.add_argument("--ask_for_gt_udf", action="store_true", help="Ask for the gt_udf name interactively if enabled")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--save_generated_udf", action="store_true", help="save generated UDF to file")

    args = parser.parse_args()
    # query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    udf_class = args.udf_class
    n_train = args.n_train
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data
    code_based = args.code_based
    model_based = args.model_based
    udf_type = "code-based" if code_based else "model-based"
    # allow_kwargs_in_udf = args.allow_kwargs_in_udf
    # num_parameter_search = args.num_parameter_search
    # labeling_budget = args.budget
    # ask_for_gt_udf = args.ask_for_gt_udf
    num_interpretations = args.num_interpretations
    save_generated_udf = args.save_generated_udf

    random.seed(run_id)
    np.random.seed(run_id)

    # input_query_file = config[dataset]["input_query_file"]
    # input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    # gt_dsl = input_query["dsl"]
    # user_query = input_query["question"]
    # positive_videos = input_query["positive_videos"]
    # y_true = [1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"])]
    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(config["log_dir"], "compare_code_and_model_udf", dataset)
    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(
        os.path.join(
            log_dir,
            (
                f"udf-{udf_class}_code-based_run-{run_id}_ninterp-{num_interpretations}.log"
                if code_based
                else f"udf-{udf_class}_model-based_run-{run_id}_ntrain-{n_train}.log"
            ),
        ),
        mode="w",
    )
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

    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )

    # registered_functions = json.load(
    #     open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r")
    # )["test"]
    registered_functions = []

    name_map = {
        "color_brown": {"signature": "Color_Brown(o0)", "description": "Whether the color of o0 is brown."},
        "color_purple": {"signature": "Color_Purple(o0)", "description": "Whether the color of o0 is purple."},
        "color_cyan": {"signature": "Color_Cyan(o0)", "description": "Whether the color of o0 is cyan."},
        "shape_cylinder": {"signature": "Shape_Cylinder(o0)", "description": "Whether the shape of o0 is cylinder."},
        "shape_cube": {"signature": "Shape_Cube(o0)", "description": "Whether the shape of o0 is cube."},
        "shape_sphere": {"signature": "Shape_Sphere(o0)", "description": "Whether the shape of o0 is sphere."},
        "material_metal": {"signature": "Material_Metal(o0)", "description": "Whether the material of o0 is metal."},
        "material_rubber": {"signature": "Material_Rubber(o0)", "description": "Whether the material of o0 is rubber."},
        "near": {"signature": "Near(o0, o1)", "description": "Whether o0 is near o1."},
        "far": {"signature": "Far(o0, o1)", "description": "Whether o0 is far away from o1."},
        "rightof": {"signature": "RightOf(o0, o1)", "description": "Whether o0 is on the right of o1."},
        "behind": {"signature": "Behind(o0, o1)", "description": "Whether o0 is behind o1."},
        "location_right": {"signature": "Location_Right(o0)", "description": "Whether o0 is on the right of the frame."},
        "location_bottom": {"signature": "Location_Bottom(o0)", "description": "Whether o0 is at the bottom of the frame."},
    }
    udf_signature = name_map[udf_class]["signature"]
    udf_description = name_map[udf_class]["description"]

    udf_name, udf_vars = parse_signature(udf_signature)
    gt_udf_name = "gt_{}.gt_0".format(udf_class)
    logger.info(f"Selected gt_udf_name: {gt_udf_name}")

    if code_based:
        up = CodeUDFWithPixelsProposer(
            config,
            prompt_config,
            None,
            registered_functions,
            dataset,
            -1,
            num_interpretations,
            None,
            -1,
            run_id,
            save_generated_udf,
            allow_kwargs_in_udf=False)
        udf_candidate_list = up.implement(udf_signature, udf_description)
        up.compute_best_test_score(gt_udf_name, udf_candidate_list)
    else:
        md = ModelDistiller(config, prompt_config, dataset, udf_signature, udf_description, gt_udf_name, run_id, n_train, save_labeled_data, load_labeled_data)
        md.prepare_data()
        md.train()
        md.test()
