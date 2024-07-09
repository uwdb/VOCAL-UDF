

import ssl
import base64
from PIL import Image, ImageFile
import http.client as httplib
import torch
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

class_names = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow']

device = "cuda" if torch.cuda.is_available() else "cpu"
final_model = torch.load('/gscratch/balazinska/enhaoz/Vehicle-Make-Color-Recognition/models/color-models/final_model_85.t', map_location=torch.device(device))
final_model.eval()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def crop_car(src_path):
    src_image = cv2.imread(src_path)
    if src_image is None:
        return
    crop_image = src_image
    dst_img = cv2.resize(src=crop_image, dsize=(224, 224))
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(dst_img)
    image = data_transforms['valid'](img).float()
    image = torch.Tensor(image)
    return image.unsqueeze(0)
#     return image.unsqueeze(0).cuda() # if cuda

def plotting(path):
    src_image = cv2.imread(path)
    plt.imshow(src_image[:,:,::-1])
    # save the plot
    plt.savefig('car_test.jpg')

def predict_color(src):
    # resp = get_box(src)
    # plotting(src)
    image = crop_car(src).to(device)
    preds = final_model(image)
    return class_names[int(preds.max(1)[1][0])]

for i in range(1, 12):
    src = f'/gscratch/balazinska/enhaoz/VOCAL-UDF/tests/car{i}.jpg'

    print("The color is {} ".format(predict_color(src)))