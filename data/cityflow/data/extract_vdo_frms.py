import os
import cv2
import argparse

import yaml


def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path


def main(data_root):
    seq_list = os.listdir(data_root)

    for seq_name in seq_list:
        if seq_name[0] != 'c':
            continue
        path_data = os.path.join(data_root, seq_name)
        path_vdo = os.path.join(path_data, 'vdo.avi')
        path_images = os.path.join(path_data, 'img1')
        if os.path.exists(path_images):
            print('Data path: %s: Already exists!' % path_data)
            continue
        check_and_create(path_images)

        vidcap = cv2.VideoCapture(path_vdo)
        success, image = vidcap.read()

        count = 1
        while success:
            path_image = os.path.join(path_images, '%06d.jpg' % count)
            cv2.imwrite(path_image, image)
            success, image = vidcap.read()
            print('Data path: %s: Frame #%06d' % (path_data, count))
            count += 1


if __name__ == '__main__':
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    data_dir = config["data_dir"]
    base_data_root = os.path.join(data_dir, 'cityflow', 'data')

    seg_list = ["train/S01", "train/S03", "train/S04", "validation/S02", "validation/S05"]
    for seg in seg_list:
        data_root = os.path.join(base_data_root, seg)
        main(data_root)
