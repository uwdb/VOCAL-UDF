from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
import duckdb
import os
import cProfile
import torch
from vocaludf import mlp
from vocaludf.utils import (
    duckdb_execute_cache_sequence,
    duckdb_execute_clevrer_cache_sequence,
    duckdb_execute_clevrer_materialize,
    parse_signature,
    PredImageDataset,
)
import torchvision.ops as ops
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image
from transformers import CLIPModel
import cv2

def py_near(img, o0_oname, o0_x1, o0_y1, o0_x2, o0_y2, o0_anames, o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o0_o1_rnames, o1_o0_rnames, height, width):
    kwargs = {"threshold": 44.50973669431999}
    threshold = kwargs.get('threshold', 50)
    center_o0 = ((o0_x1 + o0_x2) / 2, (o0_y1 + o0_y2) / 2)
    center_o1 = ((o1_x1 + o1_x2) / 2, (o1_y1 + o1_y2) / 2)
    distance = ((center_o0[0] - center_o1[0]) ** 2 + (center_o0[1] - center_o1[1]) ** 2) ** 0.5
    return distance < threshold

def foo():
    executor = ThreadPoolExecutor(max_workers=8)

    conn = duckdb.connect(
        database=os.path.join("/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir", "annotations.duckdb"),
        read_only=True,
    )

    attribute_domain = ['location_left', 'location_top', 'color_gray', 'color_red', 'color_blue', 'color_green', 'shape_cube', 'shape_sphere', 'material_rubber', 'shape_cylinder']
    attr_parameters = ','.join('?' for _ in attribute_domain)
    relationship_domain = ['left_of', 'front_of']
    rel_parameters = ','.join('?' for _ in relationship_domain)
    # Create random pandas dataframe
    sql = f"""
        CREATE TEMPORARY TABLE two_objects AS
        WITH obj_with_attrs AS (
            SELECT
                o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS attributes
            FROM clevrer_objects o
            LEFT OUTER JOIN clevrer_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid AND a.aname = ANY([{attr_parameters}])
            GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
        )
        , relationships_expanded AS (
            SELECT
                vid, fid, oid1, oid2,
                COALESCE(ARRAY_AGG(rname) FILTER (WHERE rname = ANY([{rel_parameters}])), ARRAY[]::varchar[]) AS rnames,
                ARRAY_AGG(rname) AS gt_rnames
            FROM clevrer_relationships
            GROUP BY vid, fid, oid1, oid2
        )
        SELECT
            o1.vid AS vid, o1.fid AS fid,
            o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2, o1.attributes AS o1_anames,
            o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2, o2.attributes AS o2_anames,
            COALESCE(r1.rnames, ARRAY[]::varchar[]) AS o1_o2_rnames,
            COALESCE(r2.rnames, ARRAY[]::varchar[]) AS o2_o1_rnames,
            COALESCE(r1.gt_rnames, ARRAY[]::varchar[]) AS o1_o2_gt_rnames,
            320 AS height, 480 AS width
        FROM obj_with_attrs o1
        JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid
        LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
        LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
        WHERE o1.oid <> o2.oid AND o1.vid < 1000
    """
    conn.execute(sql, attribute_domain + relationship_domain)
    df_two_objects_grouped = conn.execute("select * from two_objects").df().groupby(['vid', 'fid'])
    start = time.time()
    frames = np.random.rand(320, 480, 3)
    preds = []
    for i in tqdm(range(1000)):
        for j in range(1, 5):
            if (i, j) not in df_two_objects_grouped.groups:
                continue
            df = df_two_objects_grouped.get_group((i, j))
            frames_broadcast = np.broadcast_to(frames, (len(df), *frames.shape))
            # df['pred'] = df["vid"]
            # res = df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["height"], df["width"]
            # df_test = df[["vid", "fid", "o1_oid", "pred"]]
            preds.extend(list(executor.map(py_near, frames_broadcast, df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["o2_oname"], df["o2_x1"], df["o2_y1"], df["o2_x2"], df["o2_y2"], df["o2_anames"], df["o1_o2_rnames"], df["o2_o1_rnames"], df["height"], df["width"])))
            # df_test = df[["vid", "fid", "o1_oid", "pred"]]
    print("Time taken:", time.time() - start)
    # batch_size =
    # frames_broadcast =
    # pred = list(executor.map(udf_obj, frames_broadcast, df["o1_oname"], df["o1_x1"], df["o1_y1"], df["o1_x2"], df["o1_y2"], df["o1_anames"], df["o2_oname"], df["o2_x1"], df["o2_y1"], df["o2_x2"], df["o2_y2"], df["o2_anames"], df["o1_o2_rnames"], df["o2_o1_rnames"], df["height"], df["width"]))

def test_mlp_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    func = {
        "signature": "behind_of(o0, o1)",
        "description": "whether o0 is behind o1",
        "semantic_interpretation": "model",
        "function_implementation": "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/model_udf/clevrer/gpt4v_three_clip/behind_of/lightning_logs/version_18092606/checkpoints/udf-behind_of_run-0_ntrain-100.ckpt",
    }
    best_ckpt = func["function_implementation"]
    # Predict the labels of all the data points
    checkpoint = torch.load(best_ckpt)
    hyper_parameters = checkpoint["hyper_parameters"]
    best_mlp_model = mlp.MLPProd(**hyper_parameters)
    best_mlp_model.load_state_dict(checkpoint["state_dict"])
    best_mlp_model.eval()
    best_mlp_model.to(device)

    n_obj = 2

    conn = duckdb.connect(
        database=os.path.join("/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir", "annotations.duckdb"),
        read_only=True,
    )


    pred_dataset = PredImageDataset(conn, n_obj, "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/attribute", "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/relationship")
    pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=262144, num_workers=8, shuffle=False)

    columns = ['vid', 'fid', 'o1_oid'] if n_obj == 1 else ['vid', 'fid', 'o1_oid', 'o2_oid']
    # rows = defaultdict(list)
    rows = []
    predictions = []
    with torch.no_grad():
        for row, feature in tqdm(pred_loader):
            feature = feature.to(device)
            pred = best_mlp_model(feature)
            # for k, v in row.items():
            #     rows[k].extend(v.tolist())
            rows.extend(row.tolist())
            predictions.extend(pred.cpu().tolist())
            # logger.debug(using("profile"))
            # print("row: {}, pred: {}".format(row, pred.cpu().tolist()))
    # convert to tensor
    # df_with_pred = pd.DataFrame(rows)
    df_with_pred = pd.DataFrame(rows, columns=columns)
    df_with_pred['pred'] = predictions

def test_slow_query():
    conn = duckdb.connect(
        database=os.path.join("/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir", "annotations.duckdb"),
        read_only=True,
    )

    attribute_domain = ['location_left', 'location_top', 'color_gray', 'color_red', 'color_blue', 'color_green', 'shape_cube', 'shape_sphere', 'material_rubber', 'shape_cylinder']
    attr_parameters = ','.join('?' for _ in attribute_domain)
    sql = f"""
        CREATE TEMPORARY TABLE obj_attr_filtered AS
        SELECT
            o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
            o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
            320 AS height, 480 AS width
        FROM clevrer_objects o
        LEFT OUTER JOIN clevrer_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
        GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
    """
    conn.execute(sql, attribute_domain)

    _start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    func = {
        "signature": "material_metal(o0)",
        "description": "whether o0 is made of metal",
        "semantic_interpretation": "model",
        "function_implementation": "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/model_udf/clevrer/gpt4v_three_clip/material_metal/lightning_logs/version_18092598/checkpoints/udf-material_metal_run-0_ntrain-100.ckpt",
    }
    best_ckpt = func["function_implementation"]
    # Predict the labels of all the data points
    checkpoint = torch.load(best_ckpt)
    hyper_parameters = checkpoint["hyper_parameters"]
    best_mlp_model = mlp.MLPProd(**hyper_parameters)
    best_mlp_model.load_state_dict(checkpoint["state_dict"])
    best_mlp_model.eval()
    best_mlp_model.to(device)

    n_obj = 1

    pred_dataset = PredImageDataset(conn, n_obj, "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/attribute", "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/relationship")
    pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=262144, num_workers=8, shuffle=False)

    columns = ['vid', 'fid', 'o1_oid'] if n_obj == 1 else ['vid', 'fid', 'o1_oid', 'o2_oid']
    # rows = defaultdict(list)
    rows = []
    predictions = []
    with torch.no_grad():
        for row, feature in tqdm(pred_loader):
            feature = feature.to(device)
            pred = best_mlp_model(feature)
            # for k, v in row.items():
            #     rows[k].extend(v.tolist())
            rows.extend(row.tolist())
            predictions.extend(pred.cpu().tolist())
            # logger.debug(using("profile"))
            # print("row: {}, pred: {}".format(row, pred.cpu().tolist()))
    # convert to tensor
    # df_with_pred = pd.DataFrame(rows)
    # df_with_pred = pd.DataFrame(rows, columns=columns)
    # df_with_pred['pred'] = predictions
    # exec("material_metal = df_with_pred")
    print("Time taken:", time.time() - _start)
    print(f"# pos: {sum(predictions)}, # neg: {len(predictions) - sum(predictions)}")
    _start = time.time()
    func = {
        "signature": "behind_of(o0, o1)",
        "description": "whether o0 is behind o1",
        "semantic_interpretation": "model",
        "function_implementation": "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/model_udf/clevrer/gpt4v_three_clip/behind_of/lightning_logs/version_18092598/checkpoints/udf-behind_of_run-0_ntrain-100.ckpt",
    }
    best_ckpt = func["function_implementation"]
    # Predict the labels of all the data points
    checkpoint = torch.load(best_ckpt)
    hyper_parameters = checkpoint["hyper_parameters"]
    best_mlp_model = mlp.MLPProd(**hyper_parameters)
    best_mlp_model.load_state_dict(checkpoint["state_dict"])
    best_mlp_model.eval()
    best_mlp_model.to(device)

    n_obj = 2

    pred_dataset = PredImageDataset(conn, n_obj, "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/attribute", "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/relationship")
    pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=262144, num_workers=8, shuffle=False)

    columns = ['vid', 'fid', 'o1_oid'] if n_obj == 1 else ['vid', 'fid', 'o1_oid', 'o2_oid']
    # rows = defaultdict(list)
    rows = []
    predictions = []
    with torch.no_grad():
        for row, feature in tqdm(pred_loader):
            feature = feature.to(device)
            pred = best_mlp_model(feature)
            # for k, v in row.items():
            #     rows[k].extend(v.tolist())
            rows.extend(row.tolist())
            predictions.extend(pred.cpu().tolist())
            # logger.debug(using("profile"))
            # print("row: {}, pred: {}".format(row, pred.cpu().tolist()))
    # convert to tensor
    # df_with_pred = pd.DataFrame(rows)
    # df_with_pred = pd.DataFrame(rows, columns=columns)
    # df_with_pred['pred'] = predictions
    # exec("behind_of = df_with_pred")
    print("Time taken:", time.time() - _start)
    print(f"# pos: {sum(predictions)}, # neg: {len(predictions) - sum(predictions)}")
    # _start = time.time()
    # conn.execute("PRAGMA enable_profiling = 'json';")
    # conn.execute("PRAGMA profile_output = '/gscratch/balazinska/enhaoz/VOCAL-UDF/tests/profile.json';")
    # conn.execute("SET explain_output = 'all';")
    # res = conn.execute("""
    # EXPLAIN SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.o1_oid as o0_oid, o1.o1_oid as o1_oid, o2.o1_oid as o2_oid
    # FROM obj_attr_filtered as o0, obj_attr_filtered as o1, obj_attr_filtered as o2, behind_of as o2_behind_of_o0, material_metal as o1_material_metal
    # WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and
    #     o2.vid = o2_behind_of_o0.vid
    #     and o2.fid = o2_behind_of_o0.fid
    #     and o2.o1_oid = o2_behind_of_o0.o1_oid
    #     and o0.o1_oid = o2_behind_of_o0.o2_oid
    #     and o2_behind_of_o0.pred = 1
    #     and
    #     o1.vid = o1_material_metal.vid
    #     and o1.fid = o1_material_metal.fid
    #     and o1.o1_oid = o1_material_metal.o1_oid
    #     and o1_material_metal.pred = 1
    #     and o0.o1_oid <> o1.o1_oid and o0.o1_oid <> o2.o1_oid and o1.o1_oid <> o2.o1_oid;
    # """).fetchall()
    # print(res)
    # print("Time taken:", time.time() - _start)


# def test_slow_query_e2e():
#     conn = duckdb.connect(
#         database=os.path.join("/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir", "annotations.duckdb"),
#         read_only=True,
#     )

#     attribute_domain = ['location_left', 'location_top', 'color_gray', 'color_red', 'color_blue', 'color_green', 'shape_cube', 'shape_sphere', 'material_rubber', 'shape_cylinder']
#     attr_parameters = ','.join('?' for _ in attribute_domain)
#     sql = f"""
#         CREATE TEMPORARY TABLE obj_attr_filtered AS
#         SELECT
#             o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
#             o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
#             COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
#             COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
#             320 AS height, 480 AS width
#         FROM clevrer_objects o
#         LEFT OUTER JOIN clevrer_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
#         GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
#     """
#     conn.execute(sql, attribute_domain)

#     _start = time.time()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     func = {
#         "signature": "material_metal(o0)",
#         "description": "whether o0 is made of metal",
#         "semantic_interpretation": "model",
#         "function_implementation": "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/model_udf/clevrer/gpt4v_three_clip/material_metal/lightning_logs/version_18092598/checkpoints/udf-material_metal_run-0_ntrain-100.ckpt",
#     }
#     best_ckpt = func["function_implementation"]
#     # Predict the labels of all the data points
#     checkpoint = torch.load(best_ckpt)
#     hyper_parameters = checkpoint["hyper_parameters"]
#     best_mlp_model = mlp.MLPProd(**hyper_parameters)
#     best_mlp_model.load_state_dict(checkpoint["state_dict"])
#     best_mlp_model.eval()
#     best_mlp_model.to(device)

#     n_obj = 1

#     pred_dataset = PredImageDataset(conn, n_obj, "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/attribute", "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/relationship")
#     pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=262144, num_workers=8, shuffle=False)

#     columns = ['vid', 'fid', 'o1_oid'] if n_obj == 1 else ['vid', 'fid', 'o1_oid', 'o2_oid']
#     # rows = defaultdict(list)
#     rows = []
#     predictions = []
#     with torch.no_grad():
#         for row, feature in tqdm(pred_loader):
#             feature = feature.to(device)
#             pred = best_mlp_model(feature)
#             # for k, v in row.items():
#             #     rows[k].extend(v.tolist())
#             rows.extend(row.tolist())
#             predictions.extend(pred.cpu().tolist())
#             # logger.debug(using("profile"))
#             # print("row: {}, pred: {}".format(row, pred.cpu().tolist()))
#     # convert to tensor
#     # df_with_pred = pd.DataFrame(rows)
#     df_with_pred = pd.DataFrame(rows, columns=columns)
#     df_with_pred['pred'] = predictions
#     exec("material_metal = df_with_pred")
#     print("Time taken:", time.time() - _start)

#     _start = time.time()
#     func = {
#         "signature": "behind_of(o0, o1)",
#         "description": "whether o0 is behind o1",
#         "semantic_interpretation": "model",
#         "function_implementation": "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/model_udf/clevrer/gpt4v_three_clip/behind_of/lightning_logs/version_18092598/checkpoints/udf-behind_of_run-0_ntrain-100.ckpt",
#     }
#     best_ckpt = func["function_implementation"]
#     # Predict the labels of all the data points
#     checkpoint = torch.load(best_ckpt)
#     hyper_parameters = checkpoint["hyper_parameters"]
#     best_mlp_model = mlp.MLPProd(**hyper_parameters)
#     best_mlp_model.load_state_dict(checkpoint["state_dict"])
#     best_mlp_model.eval()
#     best_mlp_model.to(device)

#     n_obj = 2

#     pred_dataset = PredImageDataset(conn, n_obj, "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/attribute", "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/relationship")
#     pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=262144, num_workers=8, shuffle=False)

#     columns = ['vid', 'fid', 'o1_oid'] if n_obj == 1 else ['vid', 'fid', 'o1_oid', 'o2_oid']
#     # rows = defaultdict(list)
#     rows = []
#     predictions = []
#     with torch.no_grad():
#         for row, feature in tqdm(pred_loader):
#             feature = feature.to(device)
#             pred = best_mlp_model(feature)
#             # for k, v in row.items():
#             #     rows[k].extend(v.tolist())
#             rows.extend(row.tolist())
#             predictions.extend(pred.cpu().tolist())
#             # logger.debug(using("profile"))
#             # print("row: {}, pred: {}".format(row, pred.cpu().tolist()))
#     # convert to tensor
#     # df_with_pred = pd.DataFrame(rows)
#     df_with_pred = pd.DataFrame(rows, columns=columns)
#     df_with_pred['pred'] = predictions
#     exec("behind_of = df_with_pred")
#     print("Time taken:", time.time() - _start)

#     _start = time.time()
#     conn.execute("PRAGMA enable_profiling = 'json';")
#     conn.execute("PRAGMA profile_output = '/gscratch/balazinska/enhaoz/VOCAL-UDF/tests/profile.json';")
#     conn.execute("SET explain_output = 'all';")
#     res = conn.execute("""
#     EXPLAIN SELECT o0.vid as vid, o0.fid as fid, o0.o1_oid as o0_oid, o1.o1_oid as o1_oid, o2.o1_oid as o2_oid
#     FROM obj_attr_filtered as o0, obj_attr_filtered as o1, obj_attr_filtered as o2, behind_of as o2_behind_of_o0, material_metal as o1_material_metal
#     WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and
#         o2.vid = o2_behind_of_o0.vid
#         and o2.fid = o2_behind_of_o0.fid
#         and o2.o1_oid = o2_behind_of_o0.o1_oid
#         and o0.o1_oid = o2_behind_of_o0.o2_oid
#         and o2_behind_of_o0.pred = 1
#         and
#         o1.vid = o1_material_metal.vid
#         and o1.fid = o1_material_metal.fid
#         and o1.o1_oid = o1_material_metal.o1_oid
#         and o1_material_metal.pred = 1
#         and o0.o1_oid <> o1.o1_oid and o0.o1_oid <> o2.o1_oid and o1.o1_oid <> o2.o1_oid;
#     """).fetchall()
#     print(res)
#     print("Time taken:", time.time() - _start)

def bar():

    conn = duckdb.connect(
        database=os.path.join("/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir", "annotations.duckdb"),
        read_only=True,
    )

    attribute_domain = ['location_left', 'location_top', 'color_gray', 'color_red', 'color_blue', 'color_green', 'shape_cube', 'shape_sphere', 'material_rubber', 'shape_cylinder']
    attr_parameters = ','.join('?' for _ in attribute_domain)
    relationship_domain = ['left_of', 'front_of']
    rel_parameters = ','.join('?' for _ in relationship_domain)
    # Create random pandas dataframe
    sql = f"""
        WITH obj_with_attrs AS (
            SELECT
                o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS attributes
            FROM clevrer_objects o
            LEFT OUTER JOIN clevrer_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid AND a.aname = ANY([{attr_parameters}])
            GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
        )
        , relationships_expanded AS (
            SELECT
                vid, fid, oid1, oid2,
                COALESCE(ARRAY_AGG(rname) FILTER (WHERE rname = ANY([{rel_parameters}])), ARRAY[]::varchar[]) AS rnames,
                ARRAY_AGG(rname) AS gt_rnames
            FROM clevrer_relationships
            GROUP BY vid, fid, oid1, oid2
        )
        SELECT
            o1.vid AS vid, o1.fid AS fid,
            o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2, o1.attributes AS o1_anames,
            o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2, o2.attributes AS o2_anames,
            COALESCE(r1.rnames, ARRAY[]::varchar[]) AS o1_o2_rnames,
            COALESCE(r2.rnames, ARRAY[]::varchar[]) AS o2_o1_rnames,
            COALESCE(r1.gt_rnames, ARRAY[]::varchar[]) AS o1_o2_gt_rnames,
            320 AS height, 480 AS width
        FROM obj_with_attrs o1
        JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid
        LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
        LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
        WHERE o1.oid <> o2.oid AND o1.vid < 1000
    """
    start = time.time()
    df_two_objects = conn.execute(sql, attribute_domain + relationship_domain).df()
    print("Time taken:", time.time() - start)
    # start = time.time()
    # df_two_objects = conn.execute("select * from two_objects").df()
    # print("Time taken:", time.time() - start)
    start = time.time()
    df_filtered1 = df_two_objects.sample(10, random_state=0)
    print("Time taken:", time.time() - start)
    start = time.time()
    df_filtered2 = df_two_objects.sample(10, random_state=0)
    print("Time taken:", time.time() - start)

def baz():
    conn = duckdb.connect(
        database=os.path.join("/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir", "annotations.duckdb"),
        read_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    clip_model = CLIPModel.from_pretrained("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/clip-vit-base-patch32").to(device)

    attribute_domain = ['location_left', 'location_top', 'color_gray', 'color_red', 'color_blue', 'color_green', 'shape_cube', 'shape_sphere', 'material_rubber', 'shape_cylinder']
    attr_parameters = ','.join('?' for _ in attribute_domain)
    relationship_domain = ['left_of', 'front_of']
    rel_parameters = ','.join('?' for _ in relationship_domain)
    # Create random pandas dataframe
    sql = f"""
        WITH obj_with_attrs AS (
            SELECT
                o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS attributes
            FROM clevrer_objects o
            LEFT OUTER JOIN clevrer_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid AND a.aname = ANY([{attr_parameters}])
            GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
        )
        , relationships_expanded AS (
            SELECT
                vid, fid, oid1, oid2,
                COALESCE(ARRAY_AGG(rname) FILTER (WHERE rname = ANY([{rel_parameters}])), ARRAY[]::varchar[]) AS rnames,
                ARRAY_AGG(rname) AS gt_rnames
            FROM clevrer_relationships
            GROUP BY vid, fid, oid1, oid2
        )
        SELECT
            o1.vid AS vid, o1.fid AS fid,
            o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2, o1.attributes AS o1_anames,
            o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2, o2.attributes AS o2_anames,
            COALESCE(r1.rnames, ARRAY[]::varchar[]) AS o1_o2_rnames,
            COALESCE(r2.rnames, ARRAY[]::varchar[]) AS o2_o1_rnames,
            COALESCE(r1.gt_rnames, ARRAY[]::varchar[]) AS o1_o2_gt_rnames,
            320 AS height, 480 AS width
        FROM obj_with_attrs o1
        JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid
        LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
        LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
        WHERE o1.oid <> o2.oid AND o1.vid < 100
        ORDER BY o1.vid, o1.fid, o1.oid, o2.oid, o1.x1, o1.y1, o1.x2, o1.y2, o2.x1, o2.y1, o2.x2, o2.y2
    """
    start = time.time()
    df_two_objects = conn.execute(sql, attribute_domain + relationship_domain).df()
    print("Time taken:", time.time() - start)
    start = time.time()
    # df_filtered = df_two_objects.sample(500, random_state=0)
    # First 500 rows
    df_filtered = df_two_objects.iloc[:10].reset_index(drop=True)
    df_filtered['row_number'] = range(len(df_filtered))
    print("df_filtered head:", df_filtered.head())
    df_with_features = conn.execute(f"""
        SELECT any_value(d.feature) as feature
        FROM df_filtered as df,
            '/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/relationship/*.parquet' d
        WHERE df.vid=d.vid AND df.fid=d.fid
            AND df.o1_oid=d.o1_oid AND df.o2_oid=d.o2_oid
        GROUP BY df.row_number
        ORDER BY df.row_number
    """).df()
    print("Time taken:", time.time() - start)
    # convert to tensor
    features1 = torch.tensor(df_with_features['feature'].tolist())
    print("features1[0]:", features1[0])

    start = time.time()
    transforms = T.Compose([
        # T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        # T.CenterCrop(224),
        T.Lambda(lambda x: x / 255.0),        # Scale image data to [0, 1]
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    frames = []
    idxs_to_predict = []
    rois = []
    batch_boxes = []
    for i, (_, row) in enumerate(df_filtered.iterrows()):
        vid = row["vid"]
        fid = row["fid"]
        frame = frame_processing_for_program(vid, fid)
        frames.append(frame)
        o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], (320, 480))
        o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], (320, 480))
        # Verify rois are correct
        roi_x1 = min(o1_x1, o2_x1)
        roi_y1 = min(o1_y1, o2_y1)
        roi_x2 = max(o1_x2, o2_x2)
        roi_y2 = max(o1_y2, o2_y2)
        if 0 <= roi_x1 < roi_x2 and 0 <= roi_y1 < roi_y2:
            idxs_to_predict.append(i)
            rois.append([i, roi_x1, roi_y1, roi_x2, roi_y2])
            new_o1x1, new_o1y1, new_o1x2, new_o1y2, new_o2x1, new_o2y1, new_o2x2, new_o2y2 = _compute_new_box_after_crop(row, (320, 480))
            batch_boxes.append([int(new_o1x1), int(new_o1y1), int(new_o1x2), int(new_o1y2), int(new_o2x1), int(new_o2y1), int(new_o2x2), int(new_o2y2)])
    frames = np.array(frames)
    frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2).to(device) # Shape: (B, C, H, W)
    # frames = torch.stack(frames, dim=0).to(device, dtype=torch.float32) # Shape: (B, C, H, W)
    print("frames[0]: ", frames[0])
    print("idxs_to_predict:", len(idxs_to_predict))
    print("Time taken:", time.time() - start)
    _start = time.time()
    # print("frames.shape:", frames.shape)
    # print("idxs_to_predict:", idxs_to_predict)
    rois_tensor = torch.tensor(rois, dtype=torch.float).to(device)
    print("rois_tensor[0]:", rois_tensor[0])
    # https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html
    # image_patches.shape: (K, C, 224, 224), where K is the number of bounding boxes
    patch_size=(224, 224)
    image_patches = ops.roi_align(frames, rois_tensor, output_size=patch_size, spatial_scale=1.0)
    print("image_patches[0]:", image_patches[0])
    print("Time taken:", time.time() - _start)
    _start = time.time()
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
    print("batch_frames[0]: ", batch_frames[0])
    print("batch_frames_subject[0]: ", batch_frames_subject[0])
    print("batch_frames_target[0]: ", batch_frames_target[0])
    images = torch.cat([batch_frames, batch_frames_subject, batch_frames_target], dim=0) # (3N, C, H, W)
    inputs = transforms(images)
    print("inputs[0]: ", inputs[0])
    print("inputs[-1]: ", inputs[-1])
    with torch.no_grad():
        outputs = clip_model.get_image_features(pixel_values=inputs) # torch.FloatTensor of shape (3N, 512)
    features = outputs.reshape(3, N, -1).permute(1, 0, 2).reshape(N, -1) # (N, 3 * 512)
    print("features[0]: ", features[0])

    print("Time taken:", time.time() - start)
    # Move to cpu
    features2 = features.cpu()
    print("features2[0]: ", features2[0])
    for i in range(len(features1)):
        print(torch.allclose(features1[i], features2[i], atol=1e-5))
        if not torch.allclose(features1[i], features2[i], atol=1e-5):
            print(i)
            print(features1[i])
            print(features2[i])

def frame_processing_for_program(vid, fid):
    frame = np.array(Image.open(os.path.join(
            "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/video_frames",
            f"sim_{str(vid).zfill(5)}",
            f"frame_{str(fid).zfill(5)}.png"
        ))) # Shape: (H, W, C)
    return frame

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

def bazbaz():
    conn = duckdb.connect(
        database=os.path.join("/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir", "annotations.duckdb"),
        read_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    clip_model = CLIPModel.from_pretrained("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/clip-vit-base-patch32").to(device)

    attribute_domain = ['location_left', 'location_top', 'color_gray', 'color_red', 'color_blue', 'color_green', 'shape_cube', 'shape_sphere', 'material_rubber', 'shape_cylinder']
    attr_parameters = ','.join('?' for _ in attribute_domain)
    relationship_domain = ['left_of', 'front_of']
    rel_parameters = ','.join('?' for _ in relationship_domain)
    # Create random pandas dataframe
    sql = f"""
        SELECT
            o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
            o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
            320 AS height, 480 AS width
        FROM clevrer_objects o
        LEFT OUTER JOIN clevrer_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
        WHERE o.vid < 2000
        GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
        ORDER BY o.vid, o.fid, o.oid, o.x1, o.y1, o.x2, o.y2
    """
    start = time.time()
    df_one_object = conn.execute(sql, attribute_domain).df()
    print("Time taken:", time.time() - start)

    start = time.time()
    df = df_one_object.sample(2000, random_state=0)
     # extract image features
    transforms = T.Compose([
        # T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        # T.CenterCrop(224),
        T.Lambda(lambda x: x / 255.0),        # Scale image data to [0, 1]
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    frames = []
    idxs_to_predict = []
    rois = []
    for i, (_, row) in enumerate(df.iterrows()):
        vid = row["vid"]
        fid = row["fid"]
        frame = frame_processing_for_program(vid, fid)
        frames.append(frame)
        idxs_to_predict.append(i)
        rois.append([i, row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"]])
    frames = np.array(frames)
    frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2).to(device) # Shape: (B, C, H, W)

    rois_tensor = torch.tensor(rois, dtype=torch.float).to(device)
    # https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html
    # image_patches.shape: (K, C, 224, 224), where K is the number of bounding boxes
    patch_size=(224, 224)
    image_patches = ops.roi_align(frames, rois_tensor, output_size=patch_size, spatial_scale=1.0)

    # Run CLIP model
    inputs = transforms(image_patches)
    with torch.no_grad():
        features = clip_model.get_image_features(pixel_values=inputs) # torch.FloatTensor of shape (batch_size, output_dim)
    print("Time taken:", time.time() - start)
    return features

def foo1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    frame = frame_processing_for_program(0, 0)
    frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    transforms = T.Compose([
        # T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        # T.CenterCrop(224),
        T.Lambda(lambda x: x / 255.0),        # Scale image data to [0, 1]
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    ])
    inputs = transforms(frame)
    print(inputs.shape)
    clip_model = CLIPModel.from_pretrained("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/clip-vit-base-patch32").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(pixel_values=inputs)
    print(outputs)

if __name__ == '__main__':
    # cProfile.run("foo()", sort="cumtime")
    # test_mlp_predict()
    # test_slow_query()
    # baz()
    bazbaz()
    # foo1()