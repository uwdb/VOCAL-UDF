import random
import string
import os
from vocaludf.utils import (
    duckdb_execute_cache_sequence,
    duckdb_execute_clevrer_cache_sequence,
    duckdb_execute_clevrer_materialize,
    parse_signature,
    PredImageDataset,
)
from vocaludf.pretrained_model_api import image_captioning, visual_question_answering, depth_estimation
import time
import duckdb
import logging
from sklearn.metrics import f1_score
from typing import List
import torch
from vocaludf import mlp
from torch.utils.data import IterableDataset
import math
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import repeat
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

conn = duckdb.connect(
        database=os.path.join("/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb"),
        read_only=True,
    )

best_ckpt = "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/model_udf/clevrer/gpt4v_three_clip/shape_cylinder/lightning_logs/version_18416147/checkpoints/udf-shape_cylinder_run-0_ntrain-100.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Predict the labels of all the data points
checkpoint = torch.load(best_ckpt)
hyper_parameters = checkpoint["hyper_parameters"]
best_mlp_model = mlp.MLPProd(**hyper_parameters)
best_mlp_model.load_state_dict(checkpoint["state_dict"])
best_mlp_model.eval()
best_mlp_model.to(device)

signature = "shape_cylinder(o0)"
udf_name, udf_vars = parse_signature(signature)
n_obj = len(udf_vars)

pred_dataset = PredImageDataset(conn, n_obj, "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/attribute", "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/relationship")
pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=262144, num_workers=8, shuffle=False)

rows = []
predictions = []
with torch.no_grad():
    for row, feature in tqdm(pred_loader, file=sys.stdout):
        feature = feature.to(device)
        pred = best_mlp_model(feature)
        rows.extend(row.tolist())
        predictions.extend(pred.cpu().tolist())
columns = ['vid', 'fid', 'o1_oid'] if n_obj == 1 else ['vid', 'fid', 'o1_oid', 'o2_oid']
df_with_pred = pd.DataFrame(rows, columns=columns)
df_with_pred['pred'] = predictions

sql = """
    CREATE TEMPORARY TABLE one_object AS
    SELECT
        o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
        o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
        COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
        320 AS height, 480 AS width
    FROM clevrer_objects o
    LEFT OUTER JOIN clevrer_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
    GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
"""

one_object = conn.execute(sql).df()

sql = """
    SELECT df.vid AS vid, df.fid AS fid, df.o1_oid AS o1_oid, df.pred AS pred,
        CASE
            WHEN 'shape_cylinder' = ANY(o1.o1_gt_anames) THEN TRUE
            ELSE FALSE
        END AS label
    FROM df_with_pred df
    LEFT OUTER JOIN one_object o1 ON df.vid = o1.vid AND df.fid = o1.fid AND df.o1_oid = o1.o1_oid
"""

df = conn.execute(sql).df()
y_true = df['label'].tolist()
y_pred = df['pred'].tolist()
f1 = f1_score(y_true, y_pred)
print("f1 score: ", f1)
