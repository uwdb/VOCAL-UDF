import argparse
import json
import logging
import os
import random
import yaml
import numpy as np
import cv2
from PIL import Image
import duckdb
import base64
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lightning.pytorch as pl
from vocaludf import mlp
import torchmetrics
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict

config = yaml.safe_load(
    open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
)

conn = duckdb.connect(
    database=os.path.join(config["db_dir"], "annotations.duckdb"),
    read_only=True,
)

df_filtered = conn.execute("SELECT * FROM Obj_clevrer WHERE color='blue' LIMIT 1 OFFSET 400").df()
row = df_filtered.iloc[0]
print("row", row)

model_name = os.path.join(config['model_dir'], 'clip-vit-base-patch32')
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# inputs = tokenizer(["a photo of a red object", "a photo of a blue object", "a photo of a green object"], padding=True, return_tensors="pt")
# text_features = clip_model.get_text_features(**inputs)

vid = row['vid']
cap = cv2.VideoCapture(
    os.path.join(
        config['data_dir'],
        'clevrer',
        f'video_{str(vid//1000*1000).zfill(5)}-{str((vid//1000+1)*1000).zfill(5)}',
        f"video_{str(vid).zfill(5)}.mp4"
    )
)
cap.set(cv2.CAP_PROP_POS_FRAMES, row['fid'])
ret, frame = cap.read()
cap.release()
image_size = frame.shape[:2]
# x1, y1, x2, y2 = expand_box(row.x1, row.y1, row.x2, row.y2, image_size)
frame = frame[int(row['y1']):int(row['y2']), int(row['x1']):int(row['x2'])]
# resize the frame to 224x224
frame = cv2.resize(frame, (224, 224))

# inputs = processor(images=frame, return_tensors="pt").to(device)
# with torch.no_grad():
#     outputs = clip_model.get_image_features(**inputs)
# outputs = outputs.squeeze(0)
_, buffer = cv2.imencode('.jpg', frame)
base64_image = base64.b64encode(buffer).decode('utf-8')
print("base64_image", base64_image)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

inputs = processor(text=["a photo of a red object", "a photo of a blue object", "a photo of a green object"], images=frame, return_tensors="pt", padding=True)
outputs = clip_model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)
print("probs", probs)
