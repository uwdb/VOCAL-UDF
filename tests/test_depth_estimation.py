from vocaludf.pretrained_model_api import depth_estimation
from PIL import Image
import torch
import numpy as np

image = Image.open("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/gqa/images/0/10.jpg").convert("RGB")

image = np.array(image)

result = depth_estimation(image)

depth = Image.fromarray(result)

# save the image
depth.save("depth2.png")

# pip3 install torch torchvision torchaudio