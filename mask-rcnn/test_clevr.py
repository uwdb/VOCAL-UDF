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
warnings.filterwarnings('ignore')

category = True
confidence = 0.5
root = 'output'


model = torch.load( 'mask-rcnn-clevr-25.pt')
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

with open(os.path.join(root,"vocab.json"),'r') as f:
    vocab = json.load(f)
    obj2idx = vocab['object_name_to_idx']
if category:
    CLASS_NAMES = list(obj2idx.keys())
else:
    CLASS_NAMES = ['__background__', 'object']


def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img_path, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0
    
    """
    img = Image.open(img_path).convert("RGB")
    # img = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img = transform(img) #(3,240,320)
    img = img.to(device)
    print(img.shape,img.max(),type(img))
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    # print(pred_score)
    # print(len(pred_score))
    # print(max(pred_score),min(pred_score))
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    
    
    # print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


def get_prediction_category(img_path, confidence, category):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0
    
    """
    img = Image.open(img_path).convert("RGB")
    # img = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img = transform(img) #(3,240,320)
    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_labels = pred[0]['labels'][:pred_t+1]
    print(pred_class)
    print(pred_labels)
    index = pred_labels.cpu().numpy() == category
    masks = masks[index]
    pred_category_boxes = [pred_boxes[i] for i in range(len(index)) if index[i]]
    pred_category_class = [ pred_class[i] for i in range(len(index)) if index[i]]
    return masks, pred_category_boxes, pred_category_class


def segment_instance(img_path, confidence=0.5, rect_th=1, text_size=0.4, text_th=1):
    """
    segment_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    masks, boxes, pred_cls = get_prediction(img_path, confidence)
    # masks, boxes, pred_cls = get_prediction_category(img_path, confidence, 21)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for i in range(len(masks)):
      rgb_mask = get_coloured_mask(masks[i])
      img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
      cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.figure(figsize=(30,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return masks


img_path = "0.png"
plt.figure()
plt.imshow(plt.imread(img_path))

masks = segment_instance("0.png",confidence)

