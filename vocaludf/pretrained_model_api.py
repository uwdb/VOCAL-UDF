import torch
from PIL import Image
from typing import Dict, List, Union
import numpy as np
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification, BlipProcessor, BlipForQuestionAnswering, DetrImageProcessor, DetrForObjectDetection, DPTImageProcessor, DPTForDepthEstimation
import transformers
import easyocr
import io

# From https://github.com/RAIVNLab/mnms

MODEL_SELECTION = {
    "image_captioning": "Salesforce/blip-image-captioning-large",
    "image_classification": "google/vit-base-patch16-224",
    "visual_question_answering": "Salesforce/blip-vqa-base",
    "object_detection": "facebook/detr-resnet-101",
    "image_segmentation": "facebook/maskformer-swin-base-coco",
    "optical_character_recognition": "easyOCR",
    "depth_estimation": "Intel/dpt-large"
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

image_captioning_pipe = pipeline("image-to-text", model=MODEL_SELECTION["image_captioning"])
image_classification_processor = ViTImageProcessor.from_pretrained(MODEL_SELECTION["image_classification"])
image_classification_model = ViTForImageClassification.from_pretrained(MODEL_SELECTION["image_classification"])
visual_question_answering_processor = BlipProcessor.from_pretrained(MODEL_SELECTION["visual_question_answering"])
visual_question_answering_model = BlipForQuestionAnswering.from_pretrained(MODEL_SELECTION["visual_question_answering"], torch_dtype=torch.float16).to("cuda")
object_detection_processor = DetrImageProcessor.from_pretrained(MODEL_SELECTION["object_detection"], revision="no_timm")
object_detection_model = DetrForObjectDetection.from_pretrained(MODEL_SELECTION["object_detection"], revision="no_timm")
image_segmentation_feature_extractor = transformers.MaskFormerFeatureExtractor.from_pretrained(MODEL_SELECTION["image_segmentation"])
image_segmentation_model = transformers.MaskFormerForInstanceSegmentation.from_pretrained(MODEL_SELECTION["image_segmentation"]).to(
    device
)
image_segmentation_model.eval()
depth_estimation_processor = DPTImageProcessor.from_pretrained(MODEL_SELECTION["depth_estimation"])
depth_estimation_model = DPTForDepthEstimation.from_pretrained(MODEL_SELECTION["depth_estimation"]).to(device)


# -------------------------- Tool Functions --------------------------
def image_captioning(image: np.ndarray):  # alternative: nlpconnect/vit-gpt2-image-captioning (testing, blip is better than vit-gpt2)

    image = Image.fromarray(image)

    result = image_captioning_pipe(
        image
    )  # [{'generated_text': 'there is a small white dog sitting next to a cell phone'}]

    return result[0]["generated_text"]


def image_classification(image: np.ndarray):  # alternative: "microsoft/resnet-50"

    image = Image.fromarray(image)

    processor = image_classification_processor
    model = image_classification_model

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]


def visual_question_answering(image: np.ndarray, question: str):  # alternative: "dandelin/vilt-b32-finetuned-vqa"

    image = Image.fromarray(image)

    processor = visual_question_answering_processor
    model = visual_question_answering_model

    raw_image = image

    inputs = processor(raw_image, question, return_tensors="pt").to(
        "cuda", torch.float16
    )
    out = model.generate(**inputs)
    result = processor.decode(out[0], skip_special_tokens=True)

    return result


def object_detection(image: np.ndarray):  # alternative: "facebook/detr-resnet-50" # can not detect cartoon figure, might be due to the model

    image = Image.fromarray(image)

    # you can specify the revision tag if you don't want the timm dependency
    processor = object_detection_processor
    model = object_detection_model

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]
    boxes = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [int(i) for i in box.tolist()]
        boxes.append({"bbox": box, "label": model.config.id2label[label.item()]})

    return boxes


def image_segmentation(image: np.ndarray):

    img = Image.fromarray(image)

    feature_extractor = image_segmentation_feature_extractor
    model = image_segmentation_model

    inputs = feature_extractor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    outputs = feature_extractor.post_process_panoptic_segmentation(outputs)[0]
    instance_map = outputs["segmentation"].cpu().numpy()
    objs = []
    for seg in outputs["segments_info"]:
        inst_id = seg["id"]
        label_id = seg["label_id"]
        category = model.config.id2label[label_id]
        mask = (instance_map == inst_id).astype(float)
        resized_mask = np.array(
            Image.fromarray(mask).resize(img.size, resample=Image.BILINEAR)
        )
        Y, X = np.where(resized_mask > 0.5)
        x1, x2 = np.min(X), np.max(X)
        y1, y2 = np.min(Y), np.max(Y)
        num_pixels = np.sum(mask)
        objs.append(
            dict(
                mask=resized_mask,
                label=category,
                bbox=[x1, y1, x2, y2],
                inst_id=inst_id,
            )
        )

    return objs


def optical_character_recognition(image: np.ndarray):

    reader = easyocr.Reader(["en"])  # Load the OCR model into memory

    # If image is an Image object, convert it to a bytes stream
    # buffer = io.BytesIO()
    # image = Image.fromarray(image)  # Process the image if needed
    # image.save(buffer, format="JPEG")
    # buffer.seek(0)
    # image_path_or_bytes = buffer

    # Read text from the image or image path
    result = reader.readtext(image)

    # Extract only the text from the result
    result_text = [text for _, text, _ in result]

    # Format the result
    result = ", ".join(result_text)

    return result

def depth_estimation(image: np.ndarray):

    image = Image.fromarray(image)

    processor = depth_estimation_processor
    model = depth_estimation_model

    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt")
    inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    depth = (output * 255 / np.max(output)).astype("uint8")
    return depth