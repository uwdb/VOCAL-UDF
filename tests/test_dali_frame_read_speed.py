import os
import argparse
import numpy as np
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.ops as ops
import duckdb
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm
import time
import torchvision.transforms as T
from pathlib import Path
import cv2

# Only videos 0 to 9999 are used
def VideoFrameDaliDataloader(
    sequence_length=64,
    video_directory="/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/",
    device='gpu',
    batch_size=None,
    num_threads=None,
):
    assert device == 'gpu', 'dali video_resize only supports gpu backend'

    video_files = [
        os.path.join(
            video_directory,
            f"video_{str(vid//1000*1000).zfill(5)}-{str((vid//1000+1)*1000).zfill(5)}",
            f"video_{str(vid).zfill(5)}.mp4",
        )
        for vid in range(1000, 2000)
    ]

    vids = list(np.arange(1000, 2000))

    @pipeline_def
    def video_pipe(filenames, vids):
        videos, labels, start_frame_num = fn.readers.video(
            device="gpu",
            filenames=filenames,
            # the only "boosting parameter" is the sequence_length: https://github.com/NVIDIA/DALI/issues/4498
            sequence_length=sequence_length,
            pad_sequences=True,
            # shard_id=0,
            # num_shards=1,
            dtype=types.UINT8,
            random_shuffle=False,
            initial_fill=None, # Only relevant when shuffle=True
            file_list_include_preceding_frame=False, # Quiet warning about default changing
            dont_use_mmap=True,
            skip_vfr_check=True,
            enable_frame_num=True,
            labels=vids,
            name='reader',
        )
        return videos, labels, start_frame_num

    pipe = video_pipe(batch_size=batch_size, num_threads=num_threads, device_id=0, filenames=video_files, vids=vids)
    pipe.build()
    return pipe

def extract_attribute_features(sequence_length, batch_size, num_threads):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = VideoFrameDaliDataloader(sequence_length=sequence_length, batch_size=batch_size, num_threads=num_threads)

    video_iterator = DALIGenericIterator(
            [pipe],
            ['frames', 'vid', 'fid'],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            # Required or iterator loops indefinitely (https://github.com/NVIDIA/DALI/issues/2873)
            # reader_name must match name in frame::VideoFrameDaliDataloader::create_pipeline.
            reader_name='reader'
        )

    for batch in tqdm(video_iterator):
        batch = batch[0]
        # (B, 1, H, W, C) -> (B, H, W, C) -> (B, C, H, W)
        # frames = batch['frames'].squeeze(1).permute(0, 3, 1, 2)
        # vids = [v.item() for v in batch['vid'].cpu()]
        # fids = [v.item() for v in batch['fid'].cpu()]
        # (B, T, H, W, C) -> (B * T, C, H, W)
        # (B, T, H, W, C) -> (B * T, C, H, W)
        _B, _T, _H, _W, _C = batch['frames'].shape
        frames = batch['frames'].permute(0, 1, 4, 2, 3).reshape(-1, _C, _H, _W) # Shape: (B', C, H, W)


def mini_cv2_dataloader(
        batch_size: int,
        sequence_length: int,
        height=320,
        width=480,
):
    """A minimalist opencv2 dataloader
    Args:
        video_path: location of the video to be read
        batch_size: the batch dimension of the tensor to be loaded
        sequence_length: the sequence dimension of the tensor to be loaded
        height: the height (in pixels) of the video
        width: the width (in pixels) of the video
    Yields:
        A batch of video frames, stored as a pytorch tensor on the GPU in
    NOTE: This is not meant to be in any way optimised - it's the simplest way
    to load frames into a tensor (that I could think of).
    """
    video_directory="/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/"

    video_files = [
        os.path.join(
            video_directory,
            f"video_{str(vid//1000*1000).zfill(5)}-{str((vid//1000+1)*1000).zfill(5)}",
            f"video_{str(vid).zfill(5)}.mp4",
        )
        for vid in range(1000, 2000)
    ]

    next_batch = torch.zeros((batch_size, sequence_length, height, width, 3), device="cuda")
    batch_idx = 0
    seq_idx = 0
    for video_path in video_files:
        print("video_path", video_path)
        video_capture = cv2.VideoCapture(video_path)
        # next_batch = torch.zeros((batch_size, sequence_length, height, width, 3), device="cuda")
        # for batch_idx in range(batch_size):
        #     for seq_idx in range(sequence_length):
        while video_capture.isOpened():
            _, im = video_capture.read()
            if im is None:
                break
            gpu_im = torch.from_numpy(im)  # move to GPU to match DALI outputs
            next_batch[batch_idx, seq_idx] = gpu_im
            seq_idx += 1
            if seq_idx == sequence_length:
                seq_idx = 0
                batch_idx += 1
                if batch_idx == batch_size:
                    yield [{"data": next_batch}]
                    batch_idx = 0
                    next_batch = torch.zeros((batch_size, sequence_length, height, width, 3), device="cuda")

            # yield [{"data": next_batch}] # mimic DALI pipeline output

def cv2_load_frames(sequence_length, batch_size):
    dataloader = mini_cv2_dataloader(
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    for batch in tqdm(dataloader):
        shape = batch[0]["data"].shape
        # print(shape)

if __name__ == "__main__":
    # sequence_length=128, batch_size=1, num_threads=8: take about 22 mintues
    # sequence_length=128, batch_size=128, num_threads=8:
    extract_attribute_features(sequence_length=128, batch_size=128, num_threads=8)
    # cv2_load_frames(sequence_length=128, batch_size=64)