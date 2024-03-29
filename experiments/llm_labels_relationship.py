import argparse
import logging
import os
import random
import yaml
import numpy as np
from openai import OpenAI
from vocaludf.model_udf import *


client = OpenAI()

logging.basicConfig()
logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    # python llm_labels_relationship.py --run_id 0 --dataset "gqa" --relationship "on" --n_train 10 --method "balanced" --save_labeled_data
    # python llm_labels_relationship.py --run_id 0 --dataset "gqa" --relationship "on" --n_train 100 --method "balanced_three_clip" --save_labeled_data --load_labeled_data
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--relationship", type=str, help="Relationship name")
    parser.add_argument("--n_train", type=int, help="number of training samples")
    parser.add_argument("--method", type=str, help="")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data")

    args = parser.parse_args()
    run_id = args.run_id
    dataset = args.dataset
    relationship = args.relationship
    n_train = args.n_train
    method = args.method
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data

    random.seed(run_id)
    np.random.seed(run_id)

    if method == "default":
        model_distiller = GQARelationshipModelDistiller
    elif method == "balanced":
        model_distiller = GQARelationshipBalancedModelDistiller
    elif method == "balanced_clip_unnorm_bbox":
        model_distiller = GQARelationshipUnnormBboxBalancedModelDistiller
    elif method == "balanced_clip_norm_bbox":
        model_distiller = GQARelationshipNormBboxBalancedModelDistiller
    elif method == "balanced_two_clip":
        model_distiller = GQARelationshipTwoCLIPBalancedModelDistiller
    elif method == "balanced_three_clip":
        model_distiller = GQARelationshipThreeCLIPBalancedModelDistiller
    elif method == "balanced_norm_bbox_only":
        model_distiller = GQARelationshipNormBboxOnlyBalancedModelDistiller
    elif method == "llava_balanced_clip_unnorm_bbox":
        model_distiller = GQARelationshipLlavaUnnormBboxBalancedModelDistiller
    elif method == "llava_balanced_clip_norm_bbox":
        model_distiller = GQARelationshipLlavaNormBboxBalancedModelDistiller
    elif method == "llava_balanced_two_clip":
        model_distiller = GQARelationshipLlavaTwoCLIPBalancedModelDistiller
    elif method == "llava_balanced_three_clip":
        model_distiller = GQARelationshipLlavaThreeCLIPBalancedModelDistiller
    elif method == "llava_balanced_norm_bbox_only":
        model_distiller = GQARelationshipLlavaNormBboxOnlyBalancedModelDistiller
    elif method == "llava_34b_balanced_three_clip":
        model_distiller = GQARelationshipLlava34bThreeCLIPBalancedModelDistiller

    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(config["log_dir"], "llm_labels_relationship", dataset, f"{model_distiller.llm_method}_{model_distiller.mlp_method}")
    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"udf-{relationship}_run-{run_id}_ntrain-{n_train}.log"),
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

    name_map = {
        "on": {"signature": "On(o0, o1)", "description": "Whether o0 is on o1."},
        "wearing": {"signature": "Wearing(o0, o1)", "description": "Whether o0 is wearing o1."},
        "near": {"signature": "Near(o0, o1)", "description": "Whether o0 is near o1."},
        "holding": {"signature": "Holding(o0, o1)", "description": "Whether o0 is holding o1."},
        "to_the_right_of": {"signature": "to_the_right_of(o0, o1)", "description": "Whether o0 is to the right of o1."},
        "riding": {"signature": "Riding(o0, o1)", "description": "Whether o0 is riding o1."},
    }
    udf_signature = name_map[relationship]["signature"]
    udf_description = name_map[relationship]["description"]

    md = model_distiller(config, prompt_config, dataset, udf_signature, udf_description, run_id, n_train, save_labeled_data, load_labeled_data)
    md.llm_annotate_data()
    md.mlp_prepare_data()
    # lr_list = [3e-3, 3e-4]
    # n_layers = [1, 2, 3]
    # hidden_features = [16, 64, 128, 256]
    # for lr in lr_list:
    #     for n_layer in n_layers:
    #         for hidden_feature in hidden_features:
    #             mlp_config = {
    #                 "lr": lr,
    #                 "n_layers": n_layer,
    #                 "hidden_features": hidden_feature
    #             }
    md.train()
    md.test()
