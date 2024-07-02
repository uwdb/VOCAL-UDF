import random
import string
import os
from vocaludf.utils import (
    duckdb_execute_cache_sequence,
    duckdb_execute_video_materialize,
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
import yaml
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import faiss


def init_table(conn, dataset):
    # TODO: add object UDFs
    attribute_domain = []
    attr_parameters = ','.join('?' for _ in attribute_domain)
    sql = f"""
        CREATE TEMPORARY TABLE one_object AS
        SELECT
            o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
            o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
            m.height AS height, m.width AS width
        FROM {dataset}_objects o
        LEFT OUTER JOIN {dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
        JOIN {dataset}_metadata m ON o.vid = m.vid AND o.fid = m.fid
        GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2, m.height, m.width
    """
    print(f"Create one_object table:\n{sql}")
    conn.execute(sql, attribute_domain).df()

    relationship_domain = ["looking_at", "above", "in_front_of", "on_the_side_of", "carrying", "drinking_from", "have_it_on_the_back", "leaning_on", "not_contacting", "standing_on", "twisting", "wiping", "not_looking_at", "beneath", "behind", "in", "covered_by", "eating", "holding", "lying_on", "sitting_on", "touching", "wearing", "writing_on"]
    rel_parameters = ','.join('?' for _ in relationship_domain)
    sql = f"""
        WITH obj_with_attrs AS (
            SELECT
                o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS attributes
            FROM {dataset}_objects o
            LEFT OUTER JOIN {dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid AND a.aname = ANY([{attr_parameters}])
            GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
        )
        , relationships_expanded AS (
            SELECT
                vid, fid, oid1, oid2,
                COALESCE(ARRAY_AGG(rname) FILTER (WHERE rname = ANY([{rel_parameters}])), ARRAY[]::varchar[]) AS rnames,
                ARRAY_AGG(rname) AS gt_rnames
            FROM {dataset}_relationships
            GROUP BY vid, fid, oid1, oid2
        )
        SELECT
            o1.vid AS vid, o1.fid AS fid,
            o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2, o1.attributes AS o1_anames,
            o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2, o2.attributes AS o2_anames,
            COALESCE(r1.rnames, ARRAY[]::varchar[]) AS o1_o2_rnames,
            COALESCE(r2.rnames, ARRAY[]::varchar[]) AS o2_o1_rnames,
            COALESCE(r1.gt_rnames, ARRAY[]::varchar[]) AS o1_o2_gt_rnames,
            m.height AS height, m.width AS width
        FROM obj_with_attrs o1
        JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid
        JOIN {dataset}_metadata m ON o1.vid = m.vid AND o1.fid = m.fid
        LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
        LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
        WHERE o1.oid <> o2.oid
    """
    print(f"Create two_objects table:\n{sql}")
    two_objects_df = conn.execute(sql, attribute_domain + relationship_domain).df()
    return two_objects_df


def similarity_search(n_obj, num_workers, pred_batch_size, texts):
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    dataset = "charades"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conn = duckdb.connect(
        database=os.path.join(config["db_dir"], "annotations.duckdb"),
        read_only=True,
    )
    clip_model_name = os.path.join(config['model_dir'], 'clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    two_objects_df = init_table(conn, dataset)

    # get text embedding
    text_embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        text_embedding = clip_model.get_text_features(**inputs)
        # convert the embeddings to numpy array
        text_embedding = text_embedding.cpu().detach().numpy()
        text_embeddings.append(text_embedding)
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    # attribute_features_dir = os.path.join(config[dataset]["features_dir"], "attribute")
    # relationship_features_dir = os.path.join(config[dataset]["features_dir"], "relationship")
    attribute_features_dir = "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/charades_clip_with_bboxes/attribute"
    relationship_features_dir = "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/charades_clip_with_bboxes/relationship"
    pred_dataset = PredImageDataset(conn, n_obj, attribute_features_dir,relationship_features_dir)
    pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=pred_batch_size, num_workers=num_workers, shuffle=False)

    rows = []
    image_embeddings = []
    with torch.no_grad():
        for row, feature in tqdm(pred_loader):
            # ['vid', 'fid', 'o1_oid', 'o2_oid']
            for r, f in zip(row, feature):
                if r[2] == 0:
                    # convert to numpy array
                    image_embeddings.append(f.cpu().detach().numpy()[:512])
                    rows.append(r.tolist())

    print("here1")
    # convert the embeddings to numpy array
    image_embeddings = np.vstack(image_embeddings)
    image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    print("here2")
    # gpu_res = faiss.StandardGpuResources()  # use a single GPU

    # build the index: cosine similarity
    index = faiss.IndexFlatIP(512)
    # index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    print("index.is_trained", index.is_trained)
    index.add(image_embeddings)
    print("index.ntotal", index.ntotal)

    # search
    k = 100
    distances, indices = index.search(text_embeddings, k)
    print("indices: ", indices)
    print("distances: ", distances)

    # get the top k results
    # ['vid', 'fid', 'o1_oid', 'o2_oid', 'feature']
    for udf_idx in range(len(texts)):
        results = []
        for i in range(k):
            results.append(rows[indices[udf_idx][i]])

        columns = ['vid', 'fid', 'o1_oid'] if n_obj == 1 else ['vid', 'fid', 'o1_oid', 'o2_oid']
        df = pd.DataFrame(results, columns=columns)
        df["similarity"] = distances[udf_idx]
        df_out = conn.execute("""
            SELECT t1.vid AS vid, t1.fid AS fid, t1.o1_oid AS o1_oid, t1.o2_oid AS o2_oid, t2.similarity AS similarity, t1.o1_oname AS o1_oname, t1.o2_oname AS o2_oname, t1.o1_o2_gt_rnames AS o1_o2_gt_rnames
            FROM two_objects_df t1, df t2
            WHERE t1.vid = t2.vid AND t1.fid = t2.fid AND t1.o1_oid = t2.o1_oid AND t1.o2_oid = t2.o2_oid
        """).df()
        df_out.to_csv(f"clip-similarity_search-udf_idx={udf_idx}.csv", index=False)

if __name__ == "__main__":
    n_obj = 2
    num_workers = 8
    pred_batch_size = 262144

    test_inputs = [
        ["not_looking_at(o0, o1)", "Whether o0 is not looking at o1. Return True if o0 is not looking at o1, otherwise return False.", "not_looking_at"],
        ["looking_at(o0, o1)", "Whether object o0 is looking at object o1", "looking_at"],
        ["holding(o0, o1)", "Whether o0 is holding o1", "holding"],
        ["not_contacting(o0, o1)", "Whether o0 is not contacting o1. Return True if o0 is not contacting o1, otherwise return False.", "not_contacting"],
        ["touching(o0, o1)", "Whether o0 is touching o1.", "touching"],
        ["sitting_on(o0, o1)", "Whether o0 is sitting on o1.", "sitting_on"],
        ["leaning_on(o0, o1)", "Whether o0 is leaning on o1.", "leaning_on"],
        ["drinking_from(o0, o1)", "Whether o0 is drinking from o1.", "drinking_from"],
        ["standing_on(o0, o1)", "Whether o0 is standing on o1.", "standing_on"],
        ["wearing(o0, o1)", "Whether o0 is wearing o1.", "wearing"],
    ]

    # texts = [test_input[1] for test_input in test_inputs]
    texts = [test_input[2].replace("_", " ") for test_input in test_inputs]
    similarity_search(n_obj, num_workers, pred_batch_size, texts)

