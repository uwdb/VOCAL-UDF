import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import psutil
from PIL import Image

# np.random.seed(0)

def read_frame(id,vid, fid):
    # Random number from 0 to 9999
    # vid = np.random.randint(10000)
    # fid = np.random.randint(128)
    # frame = cv2.imread(
    #     os.path.join(
    #         "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/video_frames",
    #         f"sim_{str(vid).zfill(5)}",
    #         f"frame_{str(fid).zfill(5)}.png"
    #     )
    # )
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.array(
        Image.open(
            os.path.join(
                "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/video_frames",
                f"sim_{str(vid).zfill(5)}",
                f"frame_{str(fid).zfill(5)}.png",
            )
        )
    )
    return id, frame

print(psutil.cpu_count())
executor = ThreadPoolExecutor(max_workers=8)

ids = list(range(10000))
vids = np.random.randint(10000, size=10000)
fids = np.random.randint(128, size=10000)
img_col = []
id_col = []
img_mean_col = []
for result in tqdm(executor.map(read_frame, ids, vids, fids), total=len(vids)):
    id_col.append(result[0])
    img_col.append(result[1])
    img_mean_col.append(np.mean(result[1]))

# convert id_col, img_col to dataframe
df = pd.DataFrame({"id": id_col, "img": img_col, "img_mean": img_mean_col})
df = df.drop(columns=["img"])
print(df.head(10))

for img in tqdm(img_col):
    o0 = {
        'x1': 100,
        'y1': 123,
        'x2': 178,
        'y2': 345,
    }
    bbox = img[int(o0['y1']):int(o0['y2']), int(o0['x1']):int(o0['x2'])]
    avg_color = np.mean(bbox, axis=(0, 1))
    np.all(np.logical_and(avg_color >= [128, 0, 128], avg_color <= [160, 32, 160]))

executor.shutdown()