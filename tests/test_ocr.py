from vocaludf.pretrained_model_api import optical_character_recognition
from PIL import Image
import numpy as np

image = Image.open("testocr.png").convert("RGB")

image = np.array(image)

result = optical_character_recognition(image)

print(result)