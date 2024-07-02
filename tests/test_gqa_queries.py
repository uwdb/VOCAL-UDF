import json
import random
import itertools
import shutil
import numpy as np
import os
from src.utils import program_to_dsl, dsl_to_program, postgres_execute, postgres_execute_cache_sequence, print_scene_graph_helper, print_scene_graph
import csv
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
import scipy.stats as stats
import time
import psycopg2 as psycopg
import multiprocessing
from lru import LRU
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import duckdb
from vocaludf.utils import duckdb_execute_video_materialize
from duckdb_dir.udf import register_udf
import yaml

def init_table(conn, dataset, object_domain):
    obj_parameters = ','.join('?' for _ in object_domain)
    attribute_domain = ["black", "blue", "red", "large", "wood", "tall", "orange", "dark", "pink", "clear", "white", "green", "brown", "gray", "small", "yellow", "metal", "long", "silver", "standing"]
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
        WHERE o.oname = ANY([{obj_parameters}])
        GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2, m.height, m.width
    """
    print(f"Create one_object table:\n{sql}")
    conn.execute(sql, attribute_domain + object_domain).df()

    relationship_domain = ["to_the_right_of", "on", "near", "in_front_of", "next_to", "above", "below", "on_top_of", "sitting_on", "carrying", "to_the_left_of", "wearing", "of", "behind", "in", "by", "on_the_side_of", "holding", "walking_on", "beside"]
    rel_parameters = ','.join('?' for _ in relationship_domain)
    sql = f"""
        CREATE TEMPORARY TABLE two_objects AS
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
        WHERE o1.oname = ANY([{obj_parameters}]) AND o2.oname = ANY([{obj_parameters}]) AND o1.oid <> o2.oid
    """
    print(f"Create two_objects table:\n{sql}")
    conn.execute(sql, attribute_domain + relationship_domain + object_domain + object_domain).df()

config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

dataset_name = "gqa"
# query_str = "(green(o0), object(o1, 'people'), to_the_right_of(o0, o1), wearing(o1, o0))"
query_str = "(below(o0, o1), yellow(o1))"
conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=True)

obj_classes = ['window', 'man', 'shirt', 'tree', 'wall', 'person', 'building', 'ground', 'sky', 'sign', 'head', 'pole', 'hand', 'grass', 'hair', 'leg', 'car', 'woman', 'leaves', 'trees', 'table', 'ear', 'pants', 'people', 'eye', 'water', 'door', 'fence', 'nose', 'wheel', 'chair', 'floor', 'arm', 'jacket', 'hat', 'shoe', 'tail', 'clouds', 'leaf', 'face', 'letter', 'plate', 'number', 'windows', 'shorts', 'road', 'flower', 'sidewalk', 'bag', 'helmet']

init_table(conn, dataset_name, obj_classes)

input_vids = conn.execute("SELECT DISTINCT vid FROM gqa_metadata").df()["vid"].tolist()
print("Number of videos: {}".format(len(input_vids)))

program = dsl_to_program(query_str)
available_udf_names = ["to_the_right_of", "on", "near", "in_front_of", "next_to", "above", "below", "on_top_of", "sitting_on", "carrying", "black", "blue", "red", "large", "wood", "tall", "orange", "dark", "pink", "clear", "to_the_left_of", "wearing", "of", "behind", "in", "by", "on_the_side_of", "holding", "walking_on", "beside", "white", "green", "brown", "gray", "small", "yellow", "metal", "long", "silver", "standing"]
memo = [LRU(10000) for _ in range(72159)]

_start = time.time()
result, new_memo = duckdb_execute_video_materialize(conn, program, memo, None, available_udf_names, [], [])

print("Time to execute query: {}".format(time.time() - _start))
result = sorted(result)
# lock.acquire()
# for i, memo_dict in enumerate(new_memo):
#     for k, v in memo_dict.items():
#         memo[i][k] = v
# lock.release()
labels = []
for i in input_vids:
    if i in result:
        labels.append(1)
    else:
        labels.append(0)

print("Generated {} positive inputs and {} negative inputs".format(len(result), len(input_vids) - len(result)))
print("result: ", result)