import os
import argparse
import torch
# import alpha_clip
import numpy as np
import requests
from functools import partial
from io import BytesIO
from shapely.geometry import box
from typing import Dict, List, Literal, Tuple
from transformers import pipeline
from PIL import Image
# from utils import get_bbox_of_two_objects, get_img_mask_from_bbox, extract_obj_name
from torchvision import transforms
import duckdb
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from tqdm import tqdm
import pandas as pd
import yaml
import random
import string

config = yaml.safe_load(
    open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
)

def CharadesDaliDataloader(
    vids,
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
    vid_to_vname = {int(vid): vname for vname, vid in zip(df_metadata['vname'], df_metadata['vid'])}
    video_filenames = [f"{vid_to_vname[vid]}.mp4" for vid in vids]
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


# def load_image(image_file):
#     if image_file.startswith('http') or image_file.startswith('https'):
#         response = requests.get(image_file)
#         image = Image.open(BytesIO(response.content)).convert('RGB')
#     else:
#         image = Image.open(image_file).convert('RGB')

#     return image

def get_obj_center(x1, y1, x2, y2):
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    return center

# def transform_coordinates(original_coords: List[int], original_size: List[int], new_size: List[int]):
#     '''
#     transform original coordinates based on the new size
#     '''
#     original_width, original_height = original_size
#     resized_width, resized_height = new_size
#     scale_width = resized_width / original_width
#     scale_height = resized_height / original_height
#     x, y = original_coords
#     x_resized = int(x * scale_width)
#     y_resized = int(y * scale_height)
#     return x_resized, y_resized

def above(row: pd.Series):
    '''
    subject's center is above object's (i.e. subject center y < object center y)
    '''
    subj_c = get_obj_center(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'])
    obj_c = get_obj_center(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'])
    return subj_c[1] < obj_c[1]

def below(row: pd.Series):
    '''
    subject's center is below object's (i.e. subject center y > object center y)
    '''
    subj_c = get_obj_center(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'])
    obj_c = get_obj_center(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'])
    return subj_c[1] > obj_c[1]

def left(row: pd.Series):
    '''
    subject's center is to the left of object's (i.e. subject center x < object center x)
    '''
    subj_c = get_obj_center(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'])
    obj_c = get_obj_center(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'])
    return subj_c[0] < obj_c[0]

def right(row: pd.Series):
    '''
    subject's center is to the right of object's (i.e. subject center x > object center x)
    '''
    subj_c = get_obj_center(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'])
    obj_c = get_obj_center(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'])
    return subj_c[0] > obj_c[0]

def overlap(row: pd.Series):
    '''
    subject's bbox overlaps with object's
    '''
    subj_bbox = box(minx=row['o1_x1'], miny=row['o1_y1'], maxx=row['o1_x2'], maxy=row['o1_y2'])
    obj_bbox = box(minx=row['o2_x1'], miny=row['o2_y1'], maxx=row['o2_x2'], maxy=row['o2_y2'])
    return subj_bbox.intersects(obj_bbox)

def in_front_of(row: pd.Series, depth: np.array, reduce: Literal['avg', 'center'] = 'avg'):
    '''
    subject's average or center depth is > object's (greater depth values -> closer to camera)
    '''
    subj_c = get_obj_center(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'])
    obj_c = get_obj_center(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'])

    if reduce == 'avg':
        subj_depth = depth[row['o1_y1']:row['o1_y2'], row['o1_x1']:row['o1_x2']].mean()
        obj_depth = depth[row['o2_y1']:row['o2_y2'], row['o2_x1']:row['o2_x2']].mean()
    elif reduce == 'center':
        subj_depth = depth[int(subj_c[1]), int(subj_c[0])]
        obj_depth = depth[int(obj_c[1]), int(obj_c[0])]

    return subj_depth > obj_depth

def behind(row: pd.Series, depth: np.array, reduce: Literal['avg', 'center'] = 'avg'):
    '''
    subject's average or center depth is < object's (greater depth values -> closer to camera)
    '''
    subj_c = get_obj_center(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'])
    obj_c = get_obj_center(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'])

    if reduce == 'avg':
        subj_depth = depth[row['o1_y1']:row['o1_y2'], row['o1_x1']:row['o1_x2']].mean()
        obj_depth = depth[row['o2_y1']:row['o2_y2'], row['o2_x1']:row['o2_x2']].mean()
    elif reduce == 'center':
        subj_depth = depth[int(subj_c[1]), int(subj_c[0])]
        obj_depth = depth[int(obj_c[1]), int(obj_c[0])]

    return subj_depth < obj_depth

def near(row: pd.Series):
    '''
    subject's bbox is near object's bbox
    math.sqrt(math.pow(cx1 - cx2, 2.0) + math.pow(cy1 - cy2, 2.0)) / ((o1_x2 - o1_x1 + o2_x2 - o2_x1) / 2) < 1
    '''
    subj_c = get_obj_center(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'])
    obj_c = get_obj_center(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'])
    distance = np.sqrt((subj_c[0] - obj_c[0]) ** 2 + (subj_c[1] - obj_c[1]) ** 2) / ((row['o1_x2'] - row['o1_x1'] + row['o2_x2'] - row['o2_x1']) / 2)
    return distance < 1.5

def far(row: pd.Series):
    '''
    subject's bbox is far from object's bbox
    math.sqrt(math.pow(cx1 - cx2, 2.0) + math.pow(cy1 - cy2, 2.0)) / ((o1_x2 - o1_x1 + o2_x2 - o2_x1) / 2) > 1
    '''
    subj_c = get_obj_center(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'])
    obj_c = get_obj_center(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'])
    distance = np.sqrt((subj_c[0] - obj_c[0]) ** 2 + (subj_c[1] - obj_c[1]) ** 2) / ((row['o1_x2'] - row['o1_x1'] + row['o2_x2'] - row['o2_x1']) / 2)
    return distance > 3.0

def below_or_overlap(row: pd.Series):
    return below(row) or overlap(row)

def above_or_overlap(row: pd.Series):
    return above(row) or overlap(row)

class Verifier():
    def __init__(self, rules, device):
        self.rules = rules
        self.spatial_relationships = [] # Relationships(vid, fid, rid, oid1, rname, oid2)

        self.depth_array = None
        # self.depth_model = pipeline("depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device)
        self.depth_model = pipeline("depth-estimation", model=os.path.join(config['model_dir'], 'depth-anything-small-hf'), device=device)
        # self.depth_model.save_pretrained(os.path.join("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models", 'depth-anything-small-hf'))

    def compute_depth(self, image: Image.Image):
        result = self.depth_model(image)
        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            result["predicted_depth"].unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        output = prediction.squeeze().cpu().numpy()
        depth = (output * 255 / np.max(output)).astype("uint8")
        self.depth_array = depth

        # depth = Image.fromarray(depth)
        # res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
        # # save the image
        # depth.save(f"depth_{res}.png")

        self.rules.update({
            "in_front_of": partial(in_front_of, depth=self.depth_array),
            "behind": partial(behind, depth=self.depth_array),
        })

    def verify(self, row: pd.Series):
        for kw, func in self.rules.items():
            res = func(row)
            if res:
                # Relationships(vid, fid, rid, oid1, rname, oid2)
                self.spatial_relationships.append((row['vid'], row['fid'], -1, row['o1_oid'], kw, row['o2_oid']))

    def get_spatial_relationships(self):
        return self.spatial_relationships

def get_frame_vaw(vid):
    image_directory="/gscratch/balazinska/enhaoz/VOCAL-UDF/data/gqa/images"
    image_file = os.path.join(
        image_directory,
        f"{vid % 10}",
        f"{vid}.jpg",
    )
    frame = np.array(Image.open(image_file)) # Shape: (H, W, C)
    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return frame

def get_frame_cityflow(vname):
    image_directory = "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/cityflow/data"
    image_file = os.path.join(
        image_directory,
        vname
    )
    frame = np.array(Image.open(image_file)) # Shape: (H, W, C)
    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return frame

def run_charades():
    vids = list(range(9601))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conn = duckdb.connect(database="/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb", read_only=True)
    dataset = "charades"

    df_grouped = conn.execute(f"""
        SELECT o1.vid AS vid, o1.fid AS fid,
            o1.oid AS o1_oid, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
            o2.oid AS o2_oid, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2
        FROM {dataset}_objects o1, {dataset}_objects o2
        WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
        ORDER BY o1.vid, o1.fid, o1.oid, o2.oid
    """).df().groupby(['vid', 'fid'])

    # above, beneath, infrontof, behind, onthesideof, in
    rules = {
        "above": partial(above),
        "beneath": partial(below),
        "in_front_of": partial(in_front_of),
        "behind": partial(behind),
        "in": partial(overlap),
    }
    verifier = Verifier(rules, device)

    pipe = CharadesDaliDataloader(vids, sequence_length=128, batch_size=1, num_threads=1)
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

        _B, _T, _H, _W, _C = batch['frames'].shape
        # (B, 1, H, W, C) -> (B, H, W, C) -> (B, C, H, W)
        frames = batch['frames'].permute(0, 1, 4, 2, 3).reshape(-1, _C, _H, _W).to(device)
        non_zero_mask = frames.sum(dim=(1, 2, 3)) != 0
        frames = frames[non_zero_mask]
        vids = torch.repeat_interleave(batch['vid'], _T)[non_zero_mask].tolist()
        fids = (batch['fid'][:, None] + torch.arange(_T).to(device)).flatten()[non_zero_mask].tolist()

        for i in range(len(vids)):
            if (vids[i], fids[i]) not in df_grouped.groups:
                continue
            res = df_grouped.get_group((vids[i], fids[i]))
            image = transforms.functional.to_pil_image(frames[i])
            verifier.compute_depth(image)
            # NOTE: Due to data noise, multiple objects can have the same oid
            for _, row in res.iterrows():
                # Only consider spatial relationships that involve persons
                if row['o1_oid'] == 0 or row['o2_oid'] == 0:
                    verifier.verify(row)

    spatial_relationships = verifier.get_spatial_relationships()
    spatial_relationships_df = pd.DataFrame(spatial_relationships, columns=["vid", "fid", "rid", "oid1", "rname", "oid2"])
    spatial_relationships_df.to_csv("/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/charades_spatial_relationships.csv", index=False)

def run_vaw():
    dataset = "vaw"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conn = duckdb.connect(database="/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb", read_only=True)
    vids = conn.execute(f"SELECT DISTINCT vid FROM {dataset}_objects order by vid asc;").df()["vid"].tolist()



    df_grouped = conn.execute(f"""
        SELECT o1.vid AS vid, o1.fid AS fid,
            o1.oid AS o1_oid, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
            o2.oid AS o2_oid, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2
        FROM {dataset}_objects o1, {dataset}_objects o2
        WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
        ORDER BY o1.vid, o1.fid, o1.oid, o2.oid
    """).df().groupby(['vid', 'fid'])

    # above, beneath, infrontof, behind
    rules = {
        "above": partial(above),
        "beneath": partial(below),
        "to_the_left_of": partial(left),
        "to_the_right_of": partial(right),
        "in_front_of": partial(in_front_of),
        "behind": partial(behind),
    }
    verifier = Verifier(rules, device)

    for vid in tqdm(vids):
        fid = 1
        if (vid, fid) not in df_grouped.groups:
            continue
        frame = get_frame_vaw(vid) # Shape: (H, W, C) as np.array

        res = df_grouped.get_group((vid, fid))
        image = Image.fromarray(frame)
        # image = transforms.functional.to_pil_image(frame)
        verifier.compute_depth(image)
        for _, row in res.iterrows():
            verifier.verify(row)

    spatial_relationships = verifier.get_spatial_relationships()
    spatial_relationships_df = pd.DataFrame(spatial_relationships, columns=["vid", "fid", "rid", "oid1", "rname", "oid2"])
    spatial_relationships_df.to_csv(f"/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/{dataset}_spatial_relationships.csv", index=False)

def run_cityflow():
    dataset = "cityflow"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conn = duckdb.connect(database="/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb", read_only=True)
    # vids = conn.execute(f"SELECT DISTINCT vid FROM {dataset}_objects order by vid asc;").df()["vid"].tolist()
    vname_to_vid_fid = conn.execute(f"SELECT vname, vid, fid FROM {dataset}_metadata").df().set_index("vname").to_dict(orient="index")
    vnames = list(vname_to_vid_fid.keys())


    df_grouped = conn.execute(f"""
        SELECT o1.vid AS vid, o1.fid AS fid,
            o1.oid AS o1_oid, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
            o2.oid AS o2_oid, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2
        FROM {dataset}_objects o1, {dataset}_objects o2
        WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
        ORDER BY o1.vid, o1.fid, o1.oid, o2.oid
    """).df().groupby(['vid', 'fid'])

    # above, beneath, infrontof, behind
    rules = {
        "above": partial(above),
        "beneath": partial(below),
        "to_the_left_of": partial(left),
        "to_the_right_of": partial(right),
        "in_front_of": partial(in_front_of),
        "behind": partial(behind),
        # "near": partial(near),
        # "far": partial(far),
    }
    verifier = Verifier(rules, device)

    for vname in tqdm(vnames):
        vid = vname_to_vid_fid[vname]["vid"]
        fid = vname_to_vid_fid[vname]["fid"]
        if (vid, fid) not in df_grouped.groups:
            continue
        frame = get_frame_cityflow(vname) # Shape: (H, W, C) as np.array

        res = df_grouped.get_group((vid, fid))
        image = Image.fromarray(frame)
        # image = transforms.functional.to_pil_image(frame)
        verifier.compute_depth(image)
        for _, row in res.iterrows():
            verifier.verify(row)

    spatial_relationships = verifier.get_spatial_relationships()
    spatial_relationships_df = pd.DataFrame(spatial_relationships, columns=["vid", "fid", "rid", "oid1", "rname", "oid2"])
    spatial_relationships_df.to_csv(f"/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/{dataset}_spatial_relationships.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="")
    args = parser.parse_args()
    method = args.method
    if method == "charades":
        run_charades()
    elif method == "vaw":
        run_vaw()
    elif method == "cityflow":
        run_cityflow()
    else:
        raise ValueError(f"Invalid method: {method}")