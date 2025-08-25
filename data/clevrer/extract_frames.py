import os
import cv2
from tqdm import tqdm
import yaml
from multiprocessing import Pool, cpu_count

project_root = os.getenv("PROJECT_ROOT")

config = yaml.safe_load(
    open(os.path.join(project_root, "configs", "config.yaml"), "r")
)
data_dir = config["data_dir"]

frames_root = os.path.join(data_dir, "clevrer", "video_frames")
if not os.path.exists(frames_root):
    os.makedirs(frames_root)

# Create sim_00000 to sim_09999
for i in range(0, 10000):
    num = str(i).zfill(5)
    sim_dir = os.path.join(frames_root, f"sim_{num}")
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

def extract_frames(i):
    vnum = str(i).zfill(5)
    vpath = os.path.join(
        data_dir,
        'clevrer',
        f'video_{str(i//1000*1000).zfill(5)}-{str((i//1000+1)*1000).zfill(5)}',
        f"video_{vnum}.mp4"
    )
    vidcap = cv2.VideoCapture(vpath)
    success, image = vidcap.read()
    count = 0
    while success:
        c = str(count).zfill(5)
        write_path = os.path.join(frames_root, f"sim_{vnum}", f"frame_{c}.png")
        cv2.imwrite(write_path, image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()

if __name__ == "__main__":
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(extract_frames, range(0, 10000)), total=10000))