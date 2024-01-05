# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:24:25 2020

@author: zhongping zhang
"""

from dataset import *
from operator import itemgetter
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import random
import warnings
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image

warnings.filterwarnings('ignore')

confidence = 0.5
root = 'output'

model = torch.load(os.path.join(root, 'models', 'mask-rcnn-clevrer_epoch-7.pt'))
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

with open(os.path.join(root,"vocab_clevrer.json"),'r') as f:
    vocab = json.load(f)
    obj2idx = vocab['object_name_to_idx']
CLASS_NAMES = list(obj2idx.keys())

num_test = 20
vids = sorted(random.sample(range(0, 10000), num_test))
videos = [f"video_{str(i).zfill(5)}.mp4" for i in vids]
video_paths = [os.path.join(f"/home/enhao/EQUI-VOCAL/inputs/videos/video_{str(vid//1000*1000).zfill(5)}-{str((vid//1000+1)*1000).zfill(5)}", video) for vid, video in zip(vids, videos)]

for i, (vid, video_path) in enumerate(zip(vids, video_paths)):
    # read video, then sample an image from the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # random.seed(idx)
    frame_number = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, image = cap.read()
    cap.release()

    model.eval()
    with torch.no_grad():
        transform = T.Compose([T.ToTensor()])
        x = transform(image) #(3,240,320)
        x = x.to(device)
        print(x.shape,x.max(),type(x))
        pred = model([x, ])[0]
        indices = torchvision.ops.nms(pred["boxes"], pred["scores"], 0.3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = torch.from_numpy(image_rgb).permute(2, 0, 1)
    pred_labels = [f"{CLASS_NAMES[pred['labels'][idx]]}" for idx in indices]
    pred_boxes = pred["boxes"][indices].long()
    output_image = draw_bounding_boxes(tensor_image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"][indices] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5)

    # plot both tensor_image and output_image
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].imshow(tensor_image.permute(1, 2, 0))
    axs[1].imshow(output_image.permute(1, 2, 0))
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    # save the plot without coordinates
    plt.savefig(os.path.join("output", "predictions", f"{i}_output.png"), bbox_inches='tight', pad_inches=0)

