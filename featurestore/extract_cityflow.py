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
from PIL import Image
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
import math
import torchvision.transforms as T
import string
import random
import shutil

project_root = os.getenv("PROJECT_ROOT")

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

def get_frame(vname, config):
    image_directory = config["cityflow"]["video_dir"]
    image_file = os.path.join(
        image_directory,
        vname
    )
    frame = np.array(Image.open(image_file)) # Shape: (H, W, C)
    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return frame

def extract_attribute_features(conn, config, batch_size, num_threads, dataset="cityflow", patch_size=(224, 224)):
    df_grouped = conn.execute(f"""
        SELECT
            vid, fid, oid, x1, y1, x2, y2
        FROM {dataset}_objects
        ORDER BY vid, fid, oid
    """).df().groupby(['vid', 'fid'])

    clip_model_name = os.path.join(config['model_dir'], "clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

    vname_to_vid_fid = conn.execute(f"SELECT vname, vid, fid FROM {dataset}_metadata").df().set_index("vname").to_dict(orient="index")
    vnames = list(vname_to_vid_fid.keys())
    num_samples = len(vnames)

    # Copied from https://github.com/openai/CLIP/blob/main/clip/clip.py
    # Specific values from print(model.transform)
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
        ('feature', pa.list_(pa.float32())),
    ])
    # Remove existing feature files

    feature_file_dir = os.path.join(config["db_dir"], "features", f"{dataset}_three_clips", "attribute")
    if os.path.exists(feature_file_dir):
        shutil.rmtree(feature_file_dir)
    os.makedirs(feature_file_dir)

    feature_file_path = os.path.join(feature_file_dir, "0.parquet")
    partition_id = 0
    bytes_written = 0
    writer = pq.ParquetWriter(feature_file_path, schema=schema)
    for batch_start in tqdm(range(0, num_samples, batch_size)):
        batch_end = batch_start + batch_size

        batch_vnames = vnames[batch_start:batch_end]

        parquet_vids =[]
        parquet_fids = []
        parquet_oids = []
        parquet_x1s = []
        parquet_y1s = []
        parquet_x2s = []
        parquet_y2s = []
        image_patches = []
        for i in range(len(batch_vnames)):
            rois = []
            print("batch_vnames[i]", batch_vnames[i])
            vid = vname_to_vid_fid[batch_vnames[i]]["vid"]
            fid = vname_to_vid_fid[batch_vnames[i]]["fid"]
            if (vid, fid) not in df_grouped.groups:
                continue
            res = df_grouped.get_group((vid, fid))
            frame = get_frame(batch_vnames[i], config)
            _H, _W, _C = frame.shape
            # NOTE: Due to data noise, multiple objects can have the same oid
            for _, row in res.iterrows():
                x1, y1, x2, y2 = expand_box(row['x1'], row['y1'], row['x2'], row['y2'], (_H, _W), factor=1)
                rois.append([0, x1, y1, x2, y2])
                parquet_vids.append(np.uint32(vid))
                parquet_fids.append(np.uint32(fid))
                parquet_oids.append(np.uint32(row['oid']))
                parquet_x1s.append(np.uint32(row['x1']))
                parquet_y1s.append(np.uint32(row['y1']))
                parquet_x2s.append(np.uint32(row['x2']))
                parquet_y2s.append(np.uint32(row['y2']))

            single_frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) # Shape: (1, C, H, W)
            if len(rois) == 0:
                continue
            rois_tensor = torch.tensor(rois, dtype=torch.float).to(device)
            # https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html
            # image_patches.shape: (K, C, 224, 224), where K is the number of bounding boxes
            signle_frame_patches = ops.roi_align(single_frame, rois_tensor, output_size=patch_size, spatial_scale=1.0)
            image_patches.append(signle_frame_patches)

        if len(image_patches) == 0:
            continue
        image_patches = torch.cat(image_patches, dim=0)

        # Run CLIP model
        inputs = transforms(image_patches)
        with torch.no_grad():
            features = clip_model.get_image_features(pixel_values=inputs) # torch.FloatTensor of shape (batch_size, output_dim)

        # Save into Parquet file
        # Split it if too large
        features = list(features.cpu().numpy())

        if bytes_written >= 500000000: # 500MB, start a new file
            writer.close()
            partition_id += 1
            feature_file_path = os.path.join(feature_file_dir, f"{partition_id}.parquet")
            writer = pq.ParquetWriter(feature_file_path, schema=schema)
            bytes_written = 0

        batch = pa.record_batch([parquet_vids, parquet_fids, parquet_oids, parquet_x1s, parquet_y1s, parquet_x2s, parquet_y2s, features], names=['vid', 'fid', 'o1_oid', 'o1_x1', 'o1_y1', 'o1_x2', 'o1_y2', 'feature'])
        writer.write_batch(batch)
        bytes_written += batch.nbytes

def data_loader(conn, batch_size, patch_size, device, dataset="cityflow"):
    df_grouped = conn.execute(f"""
        SELECT o1.vid AS vid, o1.fid AS fid,
            o1.oid AS o1_oid, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
            o2.oid AS o2_oid, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2
        FROM {dataset}_objects o1, {dataset}_objects o2
        WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
        ORDER BY o1.vid, o1.fid, o1.oid, o2.oid
    """).df().groupby(['vid', 'fid'])

    vname_to_vid_fid = conn.execute(f"SELECT vname, vid, fid FROM {dataset}_metadata").df().set_index("vname").to_dict(orient="index")
    vnames = list(vname_to_vid_fid.keys())

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
    image_patches = []
    rois = []
    for vname in tqdm(vnames):
        vid = vname_to_vid_fid[vname]["vid"]
        fid = vname_to_vid_fid[vname]["fid"]
        if (vid, fid) not in df_grouped.groups:
            continue
        res = df_grouped.get_group((vid, fid))
        frame = get_frame(vname, config)
        _H, _W, _C = frame.shape
        # NOTE: Due to data noise, multiple objects can have the same oid
        for _, row in res.iterrows():
            o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], (_H, _W))
            o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], (_H, _W))
            # Verify rois are correct
            roi_x1 = min(o1_x1, o2_x1)
            roi_y1 = min(o1_y1, o2_y1)
            roi_x2 = max(o1_x2, o2_x2)
            roi_y2 = max(o1_y2, o2_y2)
            if 0 <= roi_x1 < roi_x2 and 0 <= roi_y1 < roi_y2:
                rois = [[0, roi_x1, roi_y1, roi_x2, roi_y2]]
                parquet_vids.append(np.uint32(vid))
                parquet_fids.append(np.uint32(fid))
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

                single_frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) # Shape: (1, C, H, W)
                rois_tensor = torch.tensor(rois, dtype=torch.float).to(device)
                image_patch = ops.roi_align(single_frame, rois_tensor, output_size=patch_size, spatial_scale=1.0)
                image_patches.append(image_patch)

                if len(image_patches) == batch_size:
                    image_patches = torch.cat(image_patches, dim=0)
                    yield parquet_vids, parquet_fids, parquet_o1_oids, parquet_o1_x1s, parquet_o1_y1s, parquet_o1_x2s, parquet_o1_y2s, parquet_o2_oids, parquet_o2_x1s, parquet_o2_y1s, parquet_o2_x2s, parquet_o2_y2s, batch_boxes, image_patches

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
                    image_patches = []
                    rois = []
    if len(image_patches) > 0:
        image_patches = torch.cat(image_patches, dim=0)
        yield parquet_vids, parquet_fids, parquet_o1_oids, parquet_o1_x1s, parquet_o1_y1s, parquet_o1_x2s, parquet_o1_y2s, parquet_o2_oids, parquet_o2_x1s, parquet_o2_y1s, parquet_o2_x2s, parquet_o2_y2s, batch_boxes, image_patches

def extract_relationship_features(conn, config, batch_size, num_threads, dataset="cityflow", patch_size=(224, 224)):
    clip_model_name = os.path.join(config['model_dir'], "clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

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

    feature_file_dir = os.path.join(config["db_dir"], "features", f"{dataset}_three_clips", "relationship")
    if os.path.exists(feature_file_dir):
        shutil.rmtree(feature_file_dir)
    os.makedirs(feature_file_dir)
    feature_file_path = os.path.join(feature_file_dir, "0.parquet")
    partition_id = 0
    bytes_written = 0
    writer = pq.ParquetWriter(feature_file_path, schema=schema)

    for data in data_loader(conn, batch_size, patch_size, device, dataset):
        parquet_vids, parquet_fids, parquet_o1_oids, parquet_o1_x1s, parquet_o1_y1s, parquet_o1_x2s, parquet_o1_y2s, parquet_o2_oids, parquet_o2_x1s, parquet_o2_y1s, parquet_o2_x2s, parquet_o2_y2s, batch_boxes, image_patches = data

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
        inputs = transforms(images)
        with torch.no_grad():
            outputs = clip_model.get_image_features(pixel_values=inputs) # torch.FloatTensor of shape (3N, 512)
        features = outputs.reshape(3, N, -1).permute(1, 0, 2).reshape(N, -1) # (N, 3 * 512)

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

def show_sequence(sequence, parquet_vids, parquet_fids, parquet_o1_oids, parquet_o2_oids=None):
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
        if parquet_o2_oids is not None:
            ax.set_title(f"vid: {parquet_vids[j]}, fid: {parquet_fids[j]}, o1_oid: {parquet_o1_oids[j]}, o2_oid: {parquet_o2_oids[j]}")
        else:
            ax.set_title(f"vid: {parquet_vids[j]}, fid: {parquet_fids[j]}, o1_oid: {parquet_o1_oids[j]}")
    # save the figure
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
    plt.savefig(f"sequence_{res}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="")
    args = parser.parse_args()
    method = args.method
    config = yaml.safe_load(
        open(os.path.join(project_root, "configs", "config.yaml"), "r")
    )
    db_dir = config["db_dir"]
    conn = duckdb.connect(database=os.path.join(db_dir, "annotations.duckdb"), read_only=True)

    print("pa.cpu_count()", pa.cpu_count())
    print("pa.io_thread_count()", pa.io_thread_count())
    if method == "attribute":
        extract_attribute_features(conn, config, batch_size=8, num_threads=1)
    elif method == "relationship":
        extract_relationship_features(conn, config, batch_size=1024, num_threads=1)
