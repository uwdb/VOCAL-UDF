from vocaludf.pretrained_model_api import image_captioning
from PIL import Image
import numpy as np
import time
image = Image.open("testocr.png").convert("RGB")

image = np.array(image)

start = time.time()
result = image_captioning(image)
print("Time taken:", time.time() - start)
print(result)