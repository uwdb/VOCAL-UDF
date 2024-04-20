from vocaludf.pretrained_model_api import depth_estimation
from PIL import Image
import torch
import numpy as np

image = Image.open("test.png").convert("RGB")

image = np.array(image)

result = depth_estimation(image)

depth = Image.fromarray(result)

# save the image
depth.save("depth.png")

# pip3 install torch torchvision torchaudio