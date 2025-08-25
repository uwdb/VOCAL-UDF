import torch
from transformers import CLIPProcessor, CLIPModel, LlavaNextForConditionalGeneration, LlavaNextProcessor
import os
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

project_root = os.getenv("PROJECT_ROOT")

config = yaml.safe_load(
    open(os.path.join(project_root, "configs", "config.yaml"), "r")
)

# Download and save the CLIP model
output_dir = os.path.join(config['model_dir'], 'clip-vit-base-patch32')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
for param in clip_model.parameters():
    param.data = param.data.contiguous()
clip_model.save_pretrained(output_dir)
clip_processor.save_pretrained(output_dir)

# Download and save the Llava model
# output_dir = os.path.join(config['model_dir'], 'llava-v1.6-34b-hf')
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# llava_model_name = "llava-hf/llava-v1.6-34b-hf"
# llava_model = LlavaNextForConditionalGeneration.from_pretrained(llava_model_name)
# llava_processor = LlavaNextProcessor.from_pretrained(llava_model_name)
# llava_model.save_pretrained(output_dir)
# llava_processor.save_pretrained(output_dir)
