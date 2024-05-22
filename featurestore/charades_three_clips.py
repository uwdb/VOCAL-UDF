import os
import argparse
import numpy as np
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, IterableDataset
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
import math
import torchvision.transforms as T
import string
import random

# class CharadesDataset(IterableDataset):
#     def __init__(self):
#         conn = duckdb.connect(database="/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb", read_only=True)
#         self.df_metadata = conn.execute(f"""
#             SELECT
#                 vname, vid, fid, width, height, split
#             FROM charades_metadata
#             ORDER BY vid, fid
#         """).df()
#         self.start = 0
#         self.end = len(self.df_metadata)
#         self.df_grouped = conn.execute(f"""
#             SELECT o1.vid AS vid, o1.fid AS fid,
#                 o1.oid AS o1_oid, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
#                 o2.oid AS o2_oid, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2
#             FROM charades_objects o1, charades_objects o2
#             WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
#             ORDER BY o1.vid, o1.fid, o1.oid, o2.oid
#         """).df().groupby(['vid', 'fid'])
#         self.root_dir = "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/charades/frames"

#     def __len__(self):
#         return len(self.df_metadata)

#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is None:  # single-process data loading, return the full iterator
#             iter_start = self.start
#             iter_end = self.end
#         else: # in a worker process
#             # split workload
#             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
#             worker_id = worker_info.id
#             iter_start = self.start + worker_id * per_worker
#             iter_end = min(iter_start + per_worker, self.end)

#         for meta_row in self.df_metadata.iloc[iter_start:iter_end]:
#             img_name = os.path.join(self.root_dir, "{}.mp4".format(meta_row['vname']), f"{str(meta_row['fid']).zfill(6)}.png")
#             image = io.imread(img_name)

#             if (meta_row['vid'], meta_row['fid']) not in self.df_grouped.groups:
#                 continue
#             res = self.df_grouped.get_group((meta_row['vid'], meta_row['fid']))
#             # NOTE: Due to data noise, multiple objects can have the same oid

#             for _, row in res.iterrows():
#                 o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], (meta_row['height'], meta_row['width']))
#                 o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], (meta_row['height'], meta_row['width']))
#                 # Verify rois are correct
#                 roi_x1 = min(o1_x1, o2_x1)
#                 roi_y1 = min(o1_y1, o2_y1)
#                 roi_x2 = max(o1_x2, o2_x2)
#                 roi_y2 = max(o1_y2, o2_y2)
#                 if 0 <= roi_x1 < roi_x2 and 0 <= roi_y1 < roi_y2:
#                     new_o1x1, new_o1y1, new_o1x2, new_o1y2, new_o2x1, new_o2y1, new_o2x2, new_o2y2 = _compute_new_box_after_crop(row, (meta_row['height'], meta_row['width']))
#                     batch_boxes.append([int(new_o1x1), int(new_o1y1), int(new_o1x2), int(new_o1y2), int(new_o2x1), int(new_o2y1), int(new_o2x2), int(new_o2y2)])

#                 yield image, row['vid'], row['fid'], row['width'], row['height']


def VideoFrameDaliDataloader(
    sequence_length=64,
    video_directory="/gscratch/balazinska/enhaoz/VOCAL-UDF/data/charades/Charades_v1_480",
    device='gpu',
    batch_size=None,
    num_threads=None,
):
    assert device == 'gpu', 'dali video_resize only supports gpu backend'
    conn = duckdb.connect(database="/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb", read_only=True)
    df_metadata = conn.execute(f"""
        SELECT DISTINCT vname, vid
        FROM charades_metadata
    """).df()
    vname_to_vid = {vname: int(vid) for vname, vid in zip(df_metadata['vname'], df_metadata['vid'])}
    video_filenames = [f"{vname}.mp4" for vname in vname_to_vid.keys()]
    video_files = [
        os.path.join(
            video_directory,
            fname,
        )
        for fname in video_filenames
    ]


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

    vids = [vname_to_vid[fname.split(".")[0]] for fname in video_filenames]
    pipe = video_pipe(batch_size=batch_size, num_threads=num_threads, device_id=0, filenames=video_files, vids=vids)
    pipe.build()
    return pipe


def expand_box(x1,y1,x2,y2,img_size,factor=1.5):
    H, W = img_size
    dw = int(factor*(x2-x1)/2)
    dh = int(factor*(y2-y1)/2)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    x1 = max(0,cx - dw)
    x2 = min(cx + dw,W)
    y1 = max(0,cy - dh)
    y2 = min(cy + dh,H)
    return [x1,y1,x2,y2]

def _compute_new_box_after_crop(row, image_size):
    o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
    o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
    x_offset = min(o1_x1, o2_x1)
    y_offset = min(o1_y1, o2_y1)
    h_ratio = 224.0 / (max(o1_y2, o2_y2) - y_offset)
    w_ratio = 224.0 / (max(o1_x2, o2_x2) - x_offset)
    return (row['o1_x1'] - x_offset) * w_ratio, (row['o1_y1'] - y_offset) * h_ratio, (row['o1_x2'] - x_offset) * w_ratio, (row['o1_y2'] - y_offset) * h_ratio, (row['o2_x1'] - x_offset) * w_ratio, (row['o2_y1'] - y_offset) * h_ratio, (row['o2_x2'] - x_offset) * w_ratio, (row['o2_y2'] - y_offset) * h_ratio

def test(batch_size, num_threads):
    _start = time.time()
    pipe = VideoFrameDaliDataloader(batch_size=batch_size, num_threads=num_threads)

    video_iterator = DALIGenericIterator(
            [pipe],
            ['data', 'label'],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            # Required or iterator loops indefinitely (https://github.com/NVIDIA/DALI/issues/2873)
            # reader_name must match name in frame::VideoFrameDaliDataloader::create_pipeline.
            reader_name='reader'
        )
    print("Time to create DALI iterator", time.time() - _start)
    for i, batch in enumerate(video_iterator):
        if i >= 10:
            break
        print("batch", batch)
        print("batch['data'].shape", batch['data'].shape)
        print("batch['label']", batch['label'])

def extract_relationship_features(conn, config, sequence_length, batch_size, num_threads, dataset="charades", patch_size=(224, 224)):
    df_grouped = conn.execute(f"""
        SELECT o1.vid AS vid, o1.fid AS fid,
            o1.oid AS o1_oid, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
            o2.oid AS o2_oid, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2
        FROM {dataset}_objects o1, {dataset}_objects o2
        WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
        ORDER BY o1.vid, o1.fid, o1.oid, o2.oid
    """).df().groupby(['vid', 'fid'])

    clip_model_name = os.path.join(config['model_dir'], "clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

    _start = time.time()
    pipe = VideoFrameDaliDataloader(sequence_length=sequence_length, batch_size=batch_size, num_threads=num_threads)

    video_iterator = DALIGenericIterator(
            [pipe],
            ['frames', 'vid', 'fid'],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            # Required or iterator loops indefinitely (https://github.com/NVIDIA/DALI/issues/2873)
            # reader_name must match name in frame::VideoFrameDaliDataloader::create_pipeline.
            reader_name='reader'
        )
    print("Time to create DALI iterator", time.time() - _start)

    transforms = T.Compose([
        # T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        # T.CenterCrop(224),
        T.Lambda(lambda x: x / 255.0),        # Scale image data to [0, 1]
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    schema = pa.schema([
        ('vid', pa.uint32()),
        ('fid', pa.uint32()),
        ('o1_oid', pa.uint32()),
        ('o1_x1', pa.uint32()),
        ('o1_y1', pa.uint32()),
        ('o1_x2', pa.uint32()),
        ('o1_y2', pa.uint32()),
        ('o2_oid', pa.uint32()),
        ('o2_x1', pa.uint32()),
        ('o2_y1', pa.uint32()),
        ('o2_x2', pa.uint32()),
        ('o2_y2', pa.uint32()),
        ('feature', pa.list_(pa.float32())),
    ])

    feature_file_dir = os.path.join(config["db_dir"], "features", "charades_three_clips", "relationship")
    os.makedirs(feature_file_dir, exist_ok=True)
    feature_file_path = os.path.join(feature_file_dir, "0.parquet")
    partition_id = 0
    bytes_written = 0
    writer = pq.ParquetWriter(feature_file_path, schema=schema)
    time4 = time.time()
    for batch in tqdm(video_iterator):
        time0 = time.time()
        print("time0", time0 - time4)
        batch = batch[0]

        # (B, 1, H, W, C) -> (B, H, W, C) -> (B, C, H, W)
        # frames = batch['frames'].squeeze(1).permute(0, 3, 1, 2)
        # vids = [v.item() for v in batch['vid'].cpu()]
        # fids = [v.item() for v in batch['fid'].cpu()]
        # (B, T, H, W, C) -> (B * T, C, H, W)
        # (B, T, H, W, C) -> (B * T, C, H, W)
        _B, _T, _H, _W, _C = batch['frames'].shape
        frames = batch['frames'].permute(0, 1, 4, 2, 3).reshape(-1, _C, _H, _W).to(device)
        non_zero_mask = frames.sum(dim=(1, 2, 3)) != 0
        frames = frames[non_zero_mask]
        vids = torch.repeat_interleave(batch['vid'], _T)[non_zero_mask].tolist()
        fids = (batch['fid'][:, None] + torch.arange(_T).to(device)).flatten()[non_zero_mask].tolist()

        rois = []
        parquet_vids =[]
        parquet_fids = []
        parquet_o1_oids = []
        parquet_o1_x1s = []
        parquet_o1_y1s = []
        parquet_o1_x2s = []
        parquet_o1_y2s = []
        parquet_o2_oids = []
        parquet_o2_x1s = []
        parquet_o2_y1s = []
        parquet_o2_x2s = []
        parquet_o2_y2s = []
        batch_boxes = []
        df_time = 0
        for i in range(len(vids)):
            df_start = time.time()
            # res = df[(df['vid'] == vids[i]) & (df['fid'] == fids[i])]
            if (vids[i], fids[i]) not in df_grouped.groups:
                continue
            res = df_grouped.get_group((vids[i], fids[i]))
            # NOTE: Due to data noise, multiple objects can have the same oid
            df_time += time.time() - df_start
            for _, row in res.iterrows():
                o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], (_H, _W))
                o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], (_H, _W))
                # Verify rois are correct
                roi_x1 = min(o1_x1, o2_x1)
                roi_y1 = min(o1_y1, o2_y1)
                roi_x2 = max(o1_x2, o2_x2)
                roi_y2 = max(o1_y2, o2_y2)
                if 0 <= roi_x1 < roi_x2 and 0 <= roi_y1 < roi_y2:
                    rois.append([i, roi_x1, roi_y1, roi_x2, roi_y2])
                    parquet_vids.append(np.uint32(vids[i]))
                    parquet_fids.append(np.uint32(fids[i]))
                    parquet_o1_oids.append(np.uint32(row['o1_oid']))
                    parquet_o1_x1s.append(np.uint32(row['o1_x1']))
                    parquet_o1_y1s.append(np.uint32(row['o1_y1']))
                    parquet_o1_x2s.append(np.uint32(row['o1_x2']))
                    parquet_o1_y2s.append(np.uint32(row['o1_y2']))
                    parquet_o2_oids.append(np.uint32(row['o2_oid']))
                    parquet_o2_x1s.append(np.uint32(row['o2_x1']))
                    parquet_o2_y1s.append(np.uint32(row['o2_y1']))
                    parquet_o2_x2s.append(np.uint32(row['o2_x2']))
                    parquet_o2_y2s.append(np.uint32(row['o2_y2']))
                    new_o1x1, new_o1y1, new_o1x2, new_o1y2, new_o2x1, new_o2y1, new_o2x2, new_o2y2 = _compute_new_box_after_crop(row, (_H, _W))
                    batch_boxes.append([int(new_o1x1), int(new_o1y1), int(new_o1x2), int(new_o1y2), int(new_o2x1), int(new_o2y1), int(new_o2x2), int(new_o2y2)])

        if len(rois) == 0:
            time4 = time.time()
            continue
        rois_tensor = torch.tensor(rois, dtype=torch.float).to(device)
        time0_5 = time.time()
        print("time0_5", time0_5 - time0)
        print("df_time", df_time)
        # https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html
        # image_patches.shape: (K, C, 224, 224), where K is the number of bounding boxes
        image_patches = ops.roi_align(frames, rois_tensor, output_size=patch_size, spatial_scale=1.0)
        # show_sequence(image_patches, parquet_vids, parquet_fids, parquet_o1_oids, parquet_o2_oids)
        time1 = time.time()
        print("time1", time1 - time0_5)

        # Run CLIP model
        batch_frames = image_patches.clone()
        # torch.tensor(image_patches).to(device)
        batch_boxes = torch.tensor(batch_boxes).to(device)
        N, C, H, W = batch_frames.shape
        X = torch.arange(W, device=device).view(1, 1, W).expand(N, H, W)
        Y = torch.arange(H, device=device).view(1, H, 1).expand(N, H, W)
        subject_masks = (X >= batch_boxes[:, 0].view(N, 1, 1).expand(N, H, W)) & (X < batch_boxes[:, 2].view(N, 1, 1).expand(N, H, W)) & (Y >= batch_boxes[:, 1].view(N, 1, 1).expand(N, H, W)) & (Y < batch_boxes[:, 3].view(N, 1, 1).expand(N, H, W))
        target_masks = (X >= batch_boxes[:, 4].view(N, 1, 1).expand(N, H, W)) & (X < batch_boxes[:, 6].view(N, 1, 1).expand(N, H, W)) & (Y >= batch_boxes[:, 5].view(N, 1, 1).expand(N, H, W)) & (Y < batch_boxes[:, 7].view(N, 1, 1).expand(N, H, W))
        batch_frames_subject = batch_frames * subject_masks.unsqueeze(1).expand(N, C, H, W)
        batch_frames_target = batch_frames * target_masks.unsqueeze(1).expand(N, C, H, W)
        images = torch.cat([batch_frames, batch_frames_subject, batch_frames_target], dim=0) # (3N, C, H, W)
        time2 = time.time()
        print("time2", time2 - time1)
        inputs = transforms(images)
        time2_5 = time.time()
        print("time2_5", time2_5 - time2)
        with torch.no_grad():
            outputs = clip_model.get_image_features(pixel_values=inputs) # torch.FloatTensor of shape (3N, 512)
        features = outputs.reshape(3, N, -1).permute(1, 0, 2).reshape(N, -1) # (N, 3 * 512)
        time3 = time.time()
        print("time3", time3 - time2_5)

        # Save into Parquet file
        # Split it if too large
        features = list(features.cpu().numpy())

        if bytes_written >= 500000000: # 500MB, start a new file
            writer.close()
            partition_id += 1
            feature_file_path = os.path.join(feature_file_dir, f"{partition_id}.parquet")
            writer = pq.ParquetWriter(feature_file_path, schema=schema)
            bytes_written = 0

        batch = pa.record_batch([parquet_vids, parquet_fids, parquet_o1_oids, parquet_o1_x1s, parquet_o1_y1s, parquet_o1_x2s, parquet_o1_y2s, parquet_o2_oids, parquet_o2_x1s, parquet_o2_y1s, parquet_o2_x2s, parquet_o2_y2s, features], names=['vid', 'fid', 'o1_oid', 'o1_x1', 'o1_y1', 'o1_x2', 'o1_y2', 'o2_oid', 'o2_x1', 'o2_y1', 'o2_x2', 'o2_y2', 'feature'])
        writer.write_batch(batch)
        bytes_written += batch.nbytes
        time4 = time.time()
        print("time4", time4 - time3)

def show_sequence(sequence, parquet_vids, parquet_fids, parquet_o1_oids, parquet_o2_oids):
    sequence = sequence.permute(0, 2, 3, 1).cpu().numpy()
    columns = 4
    sequence_length = 1
    rows = sequence.shape[0] // columns + 1
    fig = plt.figure(figsize=(32, (16 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(sequence.shape[0]):
        ax = plt.subplot(gs[j])
        ax.axis("off")
        ax.imshow(sequence[j].astype(np.uint8))
        ax.set_title(f"vid: {parquet_vids[j]}, fid: {parquet_fids[j]}, o1_oid: {parquet_o1_oids[j]}, o2_oid: {parquet_o2_oids[j]}")
    # save the figure
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
    plt.savefig(f"sequence_{res}.png")

if __name__ == "__main__":
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    conn = duckdb.connect(database="/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb", read_only=True)

    print("pa.cpu_count()", pa.cpu_count())
    print("pa.io_thread_count()", pa.io_thread_count())
    extract_relationship_features(conn, config, sequence_length=256, batch_size=1, num_threads=1)
