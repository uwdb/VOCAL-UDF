import os
import torch
from transformers import CLIPProcessor, CLIPModel
import yaml
import time
import torchvision.transforms as T

config = yaml.safe_load(
    open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
)

clip_model_name = os.path.join(config['model_dir'], "clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# random int of size (K, C, 224, 224)
images = torch.randint(0, 256, (64, 3, 224, 224)).to(device)

_start = time.time()
inputs_hf = clip_processor(images=images, return_tensors="pt").to(device)
print(f"Time taken by HF: {time.time() - _start}")
print(inputs_hf)
# Copied from https://github.com/openai/CLIP/blob/main/clip/clip.py
# Specific values from print(model.transform)
_start = time.time()
# print(images)
transforms = T.Compose([
    # T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    # T.CenterCrop(224),
    T.Lambda(lambda x: x / 255.0),        # Scale image data to [0, 1]
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])
inputs_pt = transforms(images)
print(f"Time taken by PT: {time.time() - _start}")
print(inputs_pt)

# Check if the two tensors are equal
print(torch.allclose(inputs_hf["pixel_values"], inputs_pt, atol=1e-5))
