import os
import cv2
from tqdm import tqdm

if not os.path.exists('/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/video_frames'):
    os.makedirs('/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/video_frames')

#create sim_00000 to sim_10000
for i in range(0,10000):
    num = str(i).zfill(5)
    if not os.path.exists('/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/video_frames/sim_%s'%num):
        os.makedirs('/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/video_frames/sim_%s'%num)

basepath = '/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/'

for i in tqdm(range(0,10000)):
    vnum = str(i).zfill(5)
    vpath = os.path.join(
        basepath,
        f'video_{str(i//1000*1000).zfill(5)}-{str((i//1000+1)*1000).zfill(5)}',
        f"video_{str(i).zfill(5)}.mp4"
    )
    vidcap = cv2.VideoCapture(vpath)
    success,image = vidcap.read()
    count = 0
    while success:
        c = str(count).zfill(5)
        write_path = "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/video_frames/sim_" + str(vnum) + "/frame_" + c + ".png"
        cv2.imwrite(write_path, image)
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1