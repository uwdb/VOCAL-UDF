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
    parser.add_argument("--ask_for_gt_udf", action="store_true", help="Ask for the gt_udf name interactively if enabled")
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
    ask_for_gt_udf = args.ask_for_gt_udf
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

    if code_based:
        name_map = {
            "color_brown": {"signature": "Color_Brown(o0)", "description": "Whether the color of o0 is brown."},
            "color_purple": {"signature": "Color_Purple(o0)", "description": "Whether the color of o0 is purple."},
            "shape_cylinder": {"signature": "Shape_Cylinder(o0)", "description": "Whether the shape of o0 is cylinder."},
            "material_metal": {"signature": "Material_Metal(o0)", "description": "Whether the material of o0 is metal."},
        }
        udf_signature = name_map[udf_class]["signature"]
        udf_description = name_map[udf_class]["description"]
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
        # Select the best UDF
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
        up.compute_best_test_score(gt_udf_name, udf_candidate_list)
    else:
        md = ModelDistiller(config, prompt_config, dataset, udf_class, run_id, n_train, save_labeled_data, load_labeled_data)
        md.prepare_data()
        md.train()
        md.test()
