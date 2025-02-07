import os
import cv2
from tqdm import tqdm
import yaml

config = yaml.safe_load(
    open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
)
data_dir = config["data_dir"]

if not os.path.exists(os.path.join(data_dir, "clevrer", "video_frames")):
    os.makedirs(os.path.join(data_dir, "clevrer", "video_frames"))

#create sim_00000 to sim_10000
for i in range(0,10000):
    num = str(i).zfill(5)
    if not os.path.exists(os.path.join(data_dir, "clevrer", "video_frames", f"sim_{num}")):
        os.makedirs(os.path.join(data_dir, "clevrer", "video_frames", f"sim_{num}"))

for i in tqdm(range(0,10000)):
    vnum = str(i).zfill(5)
    vpath = os.path.join(
        data_dir,
        'clevrer',
        f'video_{str(i//1000*1000).zfill(5)}-{str((i//1000+1)*1000).zfill(5)}',
        f"video_{str(i).zfill(5)}.mp4"
    )
    vidcap = cv2.VideoCapture(vpath)
    success,image = vidcap.read()
    count = 0
    while success:
        c = str(count).zfill(5)
        write_path = os.path.join(data_dir, "clevrer", "video_frames", f"sim_{vnum}", f"frame_{c}.png")
        cv2.imwrite(write_path, image)
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1