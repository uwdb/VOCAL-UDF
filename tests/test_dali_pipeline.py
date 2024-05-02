import os
import numpy as np
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.ops as ops
import duckdb
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import CLIPProcessor, CLIPModel
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm
import time
import torchvision.transforms as T

def VideoFrameDaliDataloader(
    sequence_length,
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
        for vid in range(100)
    ]

    vids = list(np.arange(100))

    @pipeline_def
    def video_pipe(filenames, vids):
        videos, labels, start_frame_num = fn.readers.video(
            device="gpu",
            filenames=filenames,
            sequence_length=sequence_length,
            pad_sequences=True,
            # shard_id=0,
            # num_shards=1,
            dtype=types.FLOAT,
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

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe1 = VideoFrameDaliDataloader(sequence_length=30, batch_size=8, num_threads=4)
pipe2 = VideoFrameDaliDataloader(sequence_length=1, batch_size=240, num_threads=4)

video_iterator1 = DALIGenericIterator(
    [pipe1],
    ['frames', 'vid', 'fid'],
    last_batch_policy=LastBatchPolicy.PARTIAL,
    reader_name='reader'
)

video_iterator2 = DALIGenericIterator(
    [pipe2],
    ['frames', 'vid', 'fid'],
    last_batch_policy=LastBatchPolicy.PARTIAL,
    reader_name='reader'
)


for i, (batch1, batch2) in enumerate(tqdm(zip(video_iterator1, video_iterator2))):
    batch1 = batch1[0]

    # (B, T, H, W, C) -> (B * T, C, H, W)
    _B, _T, _H, _W, _C = batch1['frames'].shape
    print("batch1['frames'].shape", batch1['frames'].shape)
    frames1 = batch1['frames'].permute(0, 1, 4, 2, 3).reshape(-1, _C, _H, _W).to(device)
    non_zero_mask = frames1.sum(dim=(1, 2, 3)) != 0
    frames1 = frames1[non_zero_mask]
    vids1 = torch.repeat_interleave(batch1['vid'], _T)[non_zero_mask].tolist()
    fids1 = (batch1['fid'][:, None] + torch.arange(_T).to(device)).flatten()[non_zero_mask].tolist()
    # vids1 = [v.item() for v in batch1['vid'].cpu() for _ in range(_T)]
    # fids1 = [v.item() + i for v in batch1['fid'].cpu() for i in range(_T)]


    batch2 = batch2[0]
    print("batch2['frames'].shape", batch2['frames'].shape)
    # (B, 1, H, W, C) -> (B, H, W, C) -> (B, C, H, W)
    frames2 = batch2['frames'].squeeze(1).permute(0, 3, 1, 2)
    vids2 = [v.item() for v in batch2['vid'].cpu()]
    fids2 = [v.item() for v in batch2['fid'].cpu()]

# check equality for frames, vides, fids
print("test1", torch.allclose(frames1, frames2))
print("test2", vids1 == vids2)
# print("vids1", vids1)
# print("vids2", vids2)
print("test3", fids1 == fids2)
# print("fids1", fids1)
# print("fids2", fids2)