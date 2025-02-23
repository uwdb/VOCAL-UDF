import argparse
import logging
import yaml
import os
from PIL import Image
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
from openai import OpenAI
import base64
import requests
import numpy as np
import cv2
from io import BytesIO
from vocaludf.utils import StreamToLogger, exception_hook, MODEL_COST, RESOLVE_MODEL_NAME
import sys
# https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_video(vid, prompt, pred_positive_videos, run_id, video_frames_dir, openai_model_name):
    # Load the video frames
    base64_frames = []
    for fid in range(0, 128, 4):
        image_path = os.path.join(
            video_frames_dir,
            f"sim_{str(vid).zfill(5)}",
            f"frame_{str(fid).zfill(5)}.png"
        )
        base64_frames.append(encode_image(image_path))

    request_line = {
        "custom_id": f"vid-{vid}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": openai_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *map(
                            lambda x: {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{x}"},
                            },
                            base64_frames,
                        ),
                    ],
                }
            ],
            "max_completion_tokens": 10,
            "temperature": 0.2,
            "top_p": 0.5,
            "seed": run_id,
        },
    }
    return request_line

def submit_batch(input_vids, run_id, query_id, query_filename, openai_model_name):
    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

    """
    Set up logging
    """
    base_dir = os.path.join(
        "gpt4v_clevrer_simplified",
        query_filename,
    )
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        base_dir
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(os.path.join(log_dir, "batch_tasks_qid={}-run={}.log".format(query_id, run_id)), mode="w")
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

    with open(os.path.join(config['data_dir'], "clevrer", f"{query_filename}.json"), "r") as f:
        input_query = json.load(f)['questions'][query_id]
    user_query = input_query["question"]

    prompt = f"Examine the sequence of frames from a video and determine if the following event occurs?\n'{user_query}'\nAnswer with 'yes' or 'no'."
    logger.info(f"Prompt: {prompt}")

    pred_positive_videos = []
    video_frames_dir = config["clevrer"]["video_frames_dir"]

    # Each batch file contains 10 tasks.
    def split_list(lst, size):
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    vid_chunks = split_list(input_vids, 10)
    for chunk_id, vids in tqdm(enumerate(vid_chunks)):
        logger.info(f"Processing chunk {chunk_id} with vids: {vids}")
        # 1. Preparing Your Batch File
        logger.info("Preparing Your Batch File")
        tasks = []
        for vid in vids:
            task = process_video(vid, prompt, pred_positive_videos, run_id, video_frames_dir, openai_model_name)
            tasks.append(task)
        file_name = os.path.join(log_dir, f"batch_tasks_qid={query_id}-run={run_id}-chunk={chunk_id}.jsonl")
        with open(file_name, 'w') as file:
            for obj in tasks:
                file.write(json.dumps(obj) + '\n')

        # 2. Uploading Your Batch Input File
        logger.info("Uploading Your Batch Input File")
        batch_file = client.files.create(
            file=open(file_name, "rb"),
            purpose="batch"
        )
        logger.info(f"batch_file: {batch_file}")

        # 3. Creating the Batch
        logger.info("Creating the Batch")
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        logger.info(f"batch_job: {batch_job}")
        logger.info(f"batch_job.id: {batch_job.id}")


def retrieve_batch(input_vids, run_id, query_id, query_filename, openai_model_name):
    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

    """
    Set up logging
    """
    base_dir = os.path.join(
        "gpt4v_clevrer_simplified",
        query_filename,
    )
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        base_dir
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(os.path.join(log_dir, "batch_job_results_qid={}-run={}.log".format(query_id, run_id)), mode="w")
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

    # Retrieve the batch job id from the log file
    batch_job_ids = []
    with open(os.path.join(log_dir, f"batch_tasks_qid={query_id}-run={run_id}.log"), "r") as file:
        # 2024-10-09 01:24:21,952 - vocaludf - INFO - batch_job.id: batch_67063db5c3ac819086795833bdc76d73
        lines = file.readlines()
        for line in lines:
            if "batch_job.id: " in line:
                batch_job_id = line.split("batch_job.id: ")[-1].strip()
                batch_job_ids.append(batch_job_id)

    # Check if all batch jobs are completed
    num_imcomplete_jobs = 0
    for i, batch_job_id in enumerate(batch_job_ids):
        batch_job = client.batches.retrieve(batch_job_id)
        if batch_job.status != "completed":
            num_imcomplete_jobs += 1
    if num_imcomplete_jobs > 0:
        logger.info(f"{num_imcomplete_jobs} batch jobs are not completed yet")
        return

    for chunk_idx, batch_job_id in tqdm(enumerate(batch_job_ids)):
        logger.info(f"batch_job_id: {batch_job_id}")
        batch_job = client.batches.retrieve(batch_job_id)
        logger.info(batch_job)

        result_file_id = batch_job.output_file_id
        result = client.files.content(result_file_id).content

        mode = 'wb' if chunk_idx == 0 else 'ab'
        result_file_name = os.path.join(log_dir, f"batch_job_results_qid={query_id}-run={run_id}.jsonl")

        with open(result_file_name, mode) as file:
            file.write(result)


    # Loading data from saved file
    results = []
    with open(result_file_name, 'r') as file:
        for line in file:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            results.append(json_object)

    pred_positive_videos = []
    estimated_cost = 0
    # "usage": {"prompt_tokens": 22, "completion_tokens": 2, "total_tokens": 24}, "system_fingerprint": "fp_123"}}
    for res in results:
        task_id = res['custom_id']
        # Getting index from task id
        vid = int(task_id.split('-')[-1])
        result = res['response']['body']['choices'][0]['message']['content']
        # Batch API is 50% cheaper
        estimated_cost += res['response']['body']['usage']['prompt_tokens'] * MODEL_COST[openai_model_name][0] / 2 + res['response']['body']['usage']['completion_tokens'] * MODEL_COST[openai_model_name][1] / 2
        logger.info(f"vid: {vid}, result: {result}")
        if "yes" in result.lower():
            pred_positive_videos.append(vid)

    logger.info(f"pred_positive_videos: {pred_positive_videos}")

    with open(os.path.join(config['data_dir'], "clevrer", f"{query_filename}.json"), "r") as f:
        input_query = json.load(f)['questions'][query_id]
    positive_videos = input_query["positive_videos"]

    y_true = [1 if vid in positive_videos else 0 for vid in input_vids]
    y_pred = [1 if vid in pred_positive_videos else 0 for vid in input_vids]
    logger.info(f"y_pred: {y_pred}")

    # Compute accuracy, F1, precision, recall
    f1 = f1_score(y_true, y_pred)
    logger.info(f"F1 score: {f1}")

    logger.info(f"estimated_cost: {estimated_cost}")

if __name__ == "__main__":
    # python gpt4v_clevrer_simplified.py --query_id 0 --run_id 0 --query_filename "simplified_3_new_udfs_labels" --openai_model_name "gpt-4o" --stage "submit"
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_id', type=int, help='query id')
    parser.add_argument('--run_id', type=int, help='run id')
    parser.add_argument('--query_filename', type=str, help='query filename')
    parser.add_argument("--openai_model_name", type=str, default="gpt-4-turbo-2024-04-09", help="OpenAI model name")
    parser.add_argument("--stage", type=str, help="Stage: submit or retrieve")
    args = parser.parse_args()
    query_id = args.query_id
    run_id = args.run_id
    query_filename = args.query_filename
    openai_model_name = RESOLVE_MODEL_NAME[args.openai_model_name]
    stage = args.stage

    input_vids = [i for i in range(9500, 10000)]

    if stage == "submit":
        submit_batch(input_vids, run_id, query_id, query_filename, openai_model_name)
    elif stage == "retrieve":
        retrieve_batch(input_vids, run_id, query_id, query_filename, openai_model_name)
    else:
        raise ValueError(f"Invalid stage: {stage}")