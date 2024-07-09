import argparse
import os
from vocaludf.utils import (
    replace_slot,
    parse_signature,
    PredImageDataset,
)
from vocaludf.udf_proposer import UDFProposer, UDFCandidate, CustomImageDataset
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from PIL import Image
import time
import resource
import duckdb
import logging
from openai import OpenAI
import re
from collections import defaultdict
import importlib
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import copy
import cv2
from tqdm import tqdm
import sys
from transformers import CLIPProcessor, CLIPModel, LlavaNextForConditionalGeneration, LlavaNextProcessor, CLIPTokenizer
import torch
from torch.utils.data import Dataset
import base64
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import string
import lightning.pytorch as pl
from vocaludf import mlp
import torchvision.ops as ops
import torchvision.transforms as T
import yaml
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook, get_active_domain
import shutil

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

config = yaml.safe_load(
    open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
)

def predict(dataset, udf_names, pred_batch_size, num_workers):
    conn = duckdb.connect(
        database=os.path.join(config["db_dir"], "annotations.duckdb"),
        read_only=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dataset == "charades":
        n_obj = 2
    elif dataset == "cityflow":
        n_obj = 1
    elif dataset == "clevrer":
        raise NotImplementedError

    attribute_features_dir = os.path.join(config[dataset]["features_dir"], "attribute")
    relationship_features_dir = os.path.join(config[dataset]["features_dir"], "relationship")

    rows = []
    trained_udfs = {}
    for udf_name in udf_names:
        best_ckpt = os.path.join(config["data_dir"], "trained_udfs", dataset, f"udf={udf_name}-max_positive_samples=5000.ckpt")
        # Predict the labels of all the data points
        checkpoint = torch.load(best_ckpt)
        hyper_parameters = checkpoint["hyper_parameters"]
        trained_model = mlp.MLPProd(**hyper_parameters)
        trained_model.load_state_dict(checkpoint["state_dict"])
        trained_model.eval()
        trained_model.to(device)
        trained_udfs[udf_name] = trained_model

    pred_dataset = PredImageDataset(conn, n_obj, attribute_features_dir, relationship_features_dir)
    pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=pred_batch_size, num_workers=num_workers, shuffle=False)

    # charades_relationship_predictions_df: vid, fid, rid, oid1, rname, oid2
    predictions = defaultdict(list)
    with torch.no_grad():
        for row, feature in tqdm(pred_loader, file=sys.stdout):
            feature = feature.to(device)
            rows.extend(row.tolist())
            for udf_name, trained_model in trained_udfs.items():
                pred, _ = trained_model(feature)
                predictions[udf_name].extend(pred.cpu().tolist())
    columns = ['vid', 'fid', 'oid'] if n_obj == 1 else ['vid', 'fid', 'oid1', 'oid2']
    df_with_pred = pd.DataFrame(rows, columns=columns)
    for udf_name, pred in predictions.items():
        df_with_pred[f"{udf_name}"] = pred

    # create charades_relationship_predictions dataframe
    # vid, fid, rid, oid1, rname, oid2
    # iterate through the rows of df_with_pred
    predictions_rows = []
    if dataset == "charades":
        for _, row in tqdm(df_with_pred.iterrows(), total=len(df_with_pred), file=sys.stdout, desc="create udf_predictions dataframe"):
            vid = row["vid"]
            fid = row["fid"]
            oid1 = row["oid1"]
            oid2 = row["oid2"]
            for udf_name in udf_names:
                if row[udf_name] == 1 and row['oid1'] == 0:
                    predictions_rows.append([vid, fid, -1, oid1, udf_name, oid2])

        df_predictions = pd.DataFrame(predictions_rows, columns=['vid', 'fid', 'rid', 'oid1', 'rname', 'oid2'])
        # sort
        df_predictions = df_predictions.sort_values(by=['vid', 'fid', 'oid1', 'oid2', 'rname'])
        # append charades_spatial_relationship_predictions, which is the same as charades_spatial_relationship
        df_spatial_relationships = pd.read_csv("/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/charades_spatial_relationships.csv")
        df_predictions = pd.concat([df_predictions, df_spatial_relationships], ignore_index=True)
        df_predictions.to_csv(os.path.join(config["db_dir"], "charades_relationship_predictions.csv"), index=False)
    elif dataset == "cityflow":
        for _, row in tqdm(df_with_pred.iterrows(), total=len(df_with_pred), file=sys.stdout, desc="create udf_predictions dataframe"):
            vid = row["vid"]
            fid = row["fid"]
            oid = row["oid"]
            for udf_name in udf_names:
                if row[udf_name] == 1:
                    predictions_rows.append([vid, fid, oid, udf_name])
        df_predictions = pd.DataFrame(predictions_rows, columns=['vid', 'fid', 'oid', 'aname'])
        # sort
        df_predictions = df_predictions.sort_values(by=['vid', 'fid', 'oid', 'aname'])
        df_predictions.to_csv(os.path.join(config["db_dir"], "cityflow_attribute_predictions.csv"), index=False)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # python predict_udf.py --dataset "charades" --num_workers 8 --pred_batch_size 4096
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["charades", "cityflow", "clevrer"], help="dataset name")
    parser.add_argument("--num_workers", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--pred_batch_size", type=int, default=4096, help="batch size for prediction data loader")

    args = parser.parse_args()
    dataset = args.dataset
    num_workers = args.num_workers
    pred_batch_size = args.pred_batch_size

    if dataset == "charades":
        udf_names = ["holding", "sitting_on", "standing_on", "covered_by", "carrying", "eating", "wiping", "have_it_on_the_back", "touching", "leaning_on", "wearing", "drinking_from", "lying_on", "writing_on", "twisting"]
    elif dataset == "cityflow":
        udf_names = ["suv", "white", "grey", "van", "sedan",  "black",  "red",  "blue", "pickup_truck"]
    # elif dataset == "clevrer":
    #     udf_names = ["color_gray", "color_red", "color_blue", "color_green", "shape_cube", "shape_sphere", "material_rubber", "color_brown", "color_cyan", "color_purple", "color_yellow", "shape_cylinder", "material_metal"]

    """
    Set up logging
    """
    # Create a file handler that logs even debug messages
    log_dir = os.path.join(
        config["log_dir"],
        "predict_udf",
    )
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{dataset}.log"), mode="w")
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

    predict(dataset, udf_names, pred_batch_size, num_workers)