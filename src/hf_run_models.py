import random
import torch
import json
import requests
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import ViTImageProcessor, ViTForImageClassification, DetrImageProcessor, DetrForObjectDetection, SamModel, SamProcessor
from transformers import pipeline
import os
from diffusers.utils import load_image
import matplotlib.pyplot as plt

API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_fold = "/home/enhao/JARVIS/server/models"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask

def show_masks_on_image(raw_image, masks):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
#   plt.show()
  del mask
  plt.savefig("result.jpg", bbox_inches='tight', pad_inches=0)

def text_generation(payload):
    # Example Usage:
    # data = text_generation("Can you please let us know more details about your ")
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

def image_classification(filename):
    # Example Usage:
    # data = image_classification("cats.jpg")
    image = Image.open(filename).convert("RGB")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

def object_detection(filename):
    # Example Usage:
    # data = object_detection("cats.jpg")

    detector = pipeline(task="object-detection", model=os.path.join(local_fold, "facebook/detr-resnet-101"), device=device)

    result = detector(filename)

    image = Image.open(filename).convert("RGB")
    draw = ImageDraw.Draw(image)
    for detected_object in result:
        score, label, box = detected_object["score"], detected_object["label"], [detected_object["box"]["xmin"], detected_object["box"]["ymin"], detected_object["box"]["xmax"], detected_object["box"]["ymax"]]
        draw.rectangle(box, outline=(0, 255, 0), width=2)
        font = ImageFont.truetype("arial.ttf", 16)
        draw.text((box[0], box[1]), label, font=font, fill=(0, 255, 0))

        box = [round(i, 2) for i in box]
        print(
                f"Detected {label} with confidence "
                f"{round(score, 3)} at location {box}"
        )

    # save image
    image.save("result.jpg")

def zero_shot_object_detection(filename, open_types):
    detector = pipeline(model=os.path.join(local_fold, "google/owlvit-base-patch32"), task="zero-shot-object-detection")
    # open_types = ["cat", "couch", "person", "car", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird"]

    candidate_labels = ["a photo of a {}".format(open_type) for open_type in open_types]
    result = detector(
        filename,
        candidate_labels=candidate_labels,
    )

    image = Image.open(filename).convert("RGB")
    _, image_height = image.size
    draw = ImageDraw.Draw(image)
    for detected_object in result:
        score, label, box = detected_object["score"], detected_object["label"].split("a photo of a ")[-1], [detected_object["box"]["xmin"], detected_object["box"]["ymin"], detected_object["box"]["xmax"], detected_object["box"]["ymax"]]
        draw.rectangle(box, outline=(0, 255, 0), width=2)
        font = ImageFont.truetype("arial.ttf", int(image_height * 0.05)) # means 5% of the image height

        # Calculate text size
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Calculate text position (For example, above the bounding box)
        text_x = box[0] + (box[2] - box[0] - text_width) / 2
        text_y = box[1] - text_height - 5  # 5 is just an arbitrary number for padding

        draw.text((text_x, text_y), label, font=font, fill=(0, 255, 0))
        # draw.text((box[0], box[1]), label, font=font, fill=(0, 255, 0))

        box = [round(i, 2) for i in box]
        print(
                f"Detected {label} with confidence "
                f"{round(score, 3)} at location {box}"
        )

    # save image
    image.save("result.jpg")

def image_segmentation(filename):
    # Example Usage:
    # data = image_segmentation("cats.jpg")
    pipe = pipeline(task="image-segmentation", model=os.path.join(local_fold, "facebook/detr-resnet-50-panoptic"))

    segments = pipe(filename)
    image = load_image(filename)

    colors = []
    for i in range(len(segments)):
        colors.append((random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 50))

    for i, segment in enumerate(segments):
        mask = segment["mask"]
        mask = mask.convert('L')
        layer = Image.new('RGBA', mask.size, colors[i])
        image.paste(layer, (0, 0), mask)

    # save image
    image.save("result.jpg")

def SAM(filename):
    # Segment Anything
    generator = pipeline("mask-generation", model=os.path.join(local_fold, "facebook/sam-vit-huge"), device=device)
    outputs = generator(filename, points_per_batch=64)

    masks = outputs["masks"]
    raw_image = Image.open(filename).convert("RGB")
    show_masks_on_image(raw_image, masks)

if __name__ == "__main__":
    image_classification("/home/enhao/EQUI-VOCAL/inputs/images/visualroad_1.jpg")
    # object_detection("/home/enhao/EQUI-VOCAL/inputs/images/clevrer_1.jpg")

    # zero_shot_object_detection("/home/enhao/EQUI-VOCAL/inputs/images/clevrer_1.jpg", open_types = ["blue object", "yellow object", "purple object", "cyan object"])
    # zero_shot_object_detection("/home/enhao/EQUI-VOCAL/inputs/images/visualroad_1.jpg", open_types = ["car", "bike", "person", "traffic light"])

    # image_segmentation("/home/enhao/EQUI-VOCAL/inputs/images/clevrer_1.jpg")

    # SAM("/home/enhao/EQUI-VOCAL/inputs/images/clevrer_1.jpg")

