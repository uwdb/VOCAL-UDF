import duckdb
import torch
import os
from PIL import Image
import json
import torchvision
import numpy as np
import cv2
from torchvision import transforms
from tqdm import tqdm
import yaml

# For all images in the directory 'data/', run the object detection model and store the results in a database
# Schema: Obj_clevr (fid INT, oid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float)

def normalize_coord(bbox,img_size):
    w,h = img_size
    x1,y1,x2,y2 = [int(v) for v in bbox]
    x1 = max(0,x1)
    y1 = max(0,y1)
    x2 = min(x2,w-1)
    y2 = min(y2,h-1)
    return [x1,y1,x2,y2]

if __name__ == "__main__":
    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))
    conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=False)
    conn.execute("DROP TABLE IF EXISTS Obj_clevr")
    conn.execute("CREATE TABLE Obj_clevr (fid INT, oid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float)")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clevrer_model = torch.load(os.path.join(config['data_dir'], "models", "mask-rcnn-clevrer_epoch-44.pt"))
    clevrer_model.eval()
    clevrer_model.to(device)

    with open(os.path.join(config['data_dir'], "clevr", "vocab_clevrer.json"), 'r') as f:
        vocab = json.load(f)
        obj2idx = vocab["object_name_to_idx"]
    CLASS_NAMES = list(obj2idx.keys())

    img_dir = os.path.join(config['data_dir'], "clevr/images/test")
    # 15,000 images in test set
    for fid in tqdm(range(15000)):
        filename = f"CLEVR_test_{str(fid).zfill(6)}.png"
        filepath = os.path.join(img_dir, filename)
        img = Image.open(filepath)

        with torch.no_grad():
            cv2_image = np.array(img)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            x = transform(cv2_image).to(device)
            pred = clevrer_model([x, ])[0]
            indices = torchvision.ops.nms(pred["boxes"], pred["scores"], 0.3)
        boxes = pred["boxes"][indices]
        scores = pred["scores"][indices]
        labels = pred["labels"][indices]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()
        labels = labels.cpu().detach().numpy().tolist()
        boxes = [normalize_coord(box,img.size) for box in boxes]
        objs = []
        for oid, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            color, material, shape = CLASS_NAMES[label].split(" ")
            # insert into database
            conn.execute("INSERT INTO Obj_clevr VALUES (?,?,?,?,?,?,?,?,?)", [fid, oid, shape, color, material, box[0], box[1], box[2], box[3]])
        conn.execute("SELECT * FROM Obj_clevr")
    # print(conn.fetchall())
    conn.close()