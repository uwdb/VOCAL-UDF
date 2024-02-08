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

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ccot', action='store_true', help='use ccot technique')
    parser.add_argument('--save_output', action='store_true', help='save the output')
    parser.add_argument('--output_dir', type=str, help='where to save the results')
    parser.add_argument('--run_id', type=int, help='run id')
    parser.add_argument('--question_id', type=int, help='question id')
    parser.add_argument('--task_name', type=str, help='task name')
    args = parser.parse_args()
    use_ccot = args.use_ccot
    save_output = args.save_output
    output_dir = args.output_dir
    run_id = args.run_id
    question_id = args.question_id
    task_name = args.task_name

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

    num_images = 20

    # read json file
    with open(os.path.join(config['data_dir'], "clevr", "{}.json".format(task_name)), "r") as f:
        data = json.load(f)
    question = data['questions'][question_id]['question']
    gt_positive_images = data['questions'][question_id]['positive_images']
    if use_ccot:
        s_in = """For the provided image and its associated question, generate a scene graph in JSON format that includes the following: 1. Objects that are relevant to answering the question. 2. Object attributes that are relevant to answering the question. 3. Object relationships that are relevant to answering the question. Scene Graph:
        """
        p_in = f"Question: \"Is the following description correct for this image? {question}\""
        stage1_prompt = f"{p_in}\n{s_in}"
        context = "Use the image and scene graph as context and answer the following question:"
        print("stage1_prompt:", stage1_prompt)
    else:
        prompt = f"Is the following description correct for this image? {question} Answer with 'yes' or 'no'."
        print(prompt)

    gt_labels = []
    for fid in range(num_images):
        if fid in gt_positive_images:
            gt_labels.append(1)
        else:
            gt_labels.append(0)

    pred_positive_images = []
    img_dir = os.path.join(config['data_dir'], "clevr/images/test")

    for i in tqdm(range(0, num_images)):
        filename = f"CLEVR_test_{str(i).zfill(6)}.png"
        filepath = os.path.join(img_dir, filename)
        base64_image = encode_image(filepath)
        if use_ccot:
            # Stage 1
            stage1_prompt
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": stage1_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.2,
                top_p=0.5,
                seed=run_id
            )
            result = response.choices[0].message.content
            # Stage 2
            s_g = f"Scene Graph:\n{result}"
            stage2_prompt = f"{s_g}\n{context}\n{p_in}\nAnswer with 'yes' or 'no'."
            print("stage2_prompt:", stage2_prompt)
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": stage2_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=10,
                temperature=0.2,
                top_p=0.5,
                seed=run_id
            )
            result = response.choices[0].message.content
            print(result)
            if "yes" in result.lower():
                pred_positive_images.append(i)
        else:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=10,
                temperature=0.2,
                top_p=0.5,
                seed=run_id
            )
            result = response.choices[0].message.content
            print(result)
            if "yes" in result.lower():
                pred_positive_images.append(i)
    print(pred_positive_images)

    pred_labels = []
    for fid in range(num_images):
        if fid in pred_positive_images:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

    # Compute accuracy, F1, precision, recall
    accuracy = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    if save_output:
        # Save the output
        output = {
            "question": question,
            "gt_positive_images": gt_positive_images,
            "pred_positive_images": pred_positive_images,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        # create output directory if not exists
        dir_name = "with_ccot" if use_ccot else "without_ccot"
        if not os.path.exists(os.path.join(output_dir, 'gpt4v_clevr', dir_name)):
            os.makedirs(os.path.join(output_dir, 'gpt4v_clevr', dir_name))

        with open(os.path.join(output_dir, 'gpt4v_clevr', dir_name, f"task_{task_name}_run_{run_id}_question_{question_id}.json"), "w") as f:
            json.dump(output, f)
