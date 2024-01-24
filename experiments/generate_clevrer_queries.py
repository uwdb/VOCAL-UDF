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
from vocaludf.utils import duckdb_execute_cache_sequence, duckdb_execute_clevrer_cache_sequence
from duckdb_dir.udf import register_udf
import yaml

random.seed(1234)
# np.random.seed(10)

m = multiprocessing.Manager()
lock = m.Lock()
memo = [LRU(10000) for _ in range(72159)]

def generate_clevrer_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nattr_pred, nvars, depth, max_duration, nunsupported_udfs, supported_list, supported_attr_list, unsupported_list, unsupported_attr_list, max_workers, dataset_name, config):
    """
    Generate (n_queries) queries with (npred) predicates and (nvars) variables. Each predicate is randomly chosen from (supported_list) and (nunsupported_udfs) unsupported UDFs from(unsupported_list), where each unsupported UDF must be used at least once.
    """

    conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=True)
    # conn = duckdb.connect()
    # conn.execute("CREATE TABLE Obj_clevrer (oid INT, vid INT, fid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float)")
    # conn.execute("COPY Obj_clevrer FROM '{}' (FORMAT 'csv', delimiter ',', header 0)".format(os.path.join(config["db_dir"], "obj_clevrer.csv")))
    # Creating index seems to produce incorrect results
    # conn.execute("CREATE INDEX IF NOT EXISTS idx_obj_clevrer ON Obj_clevrer (vid)")
    register_udf(conn)
    queries = []
    while len(queries) < n_queries:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(generate_one_query, repeat(conn, max_workers), repeat(ratio_lower_bound, max_workers), repeat(ratio_upper_bound, max_workers), repeat(npred, max_workers), repeat(nattr_pred, max_workers), repeat(nvars, max_workers), repeat(depth, max_workers), repeat(max_duration, max_workers), repeat(nunsupported_udfs, max_workers), repeat(supported_list, max_workers), repeat(supported_attr_list, max_workers), repeat(unsupported_list, max_workers), repeat(unsupported_attr_list, max_workers), repeat(dataset_name, max_workers)):
                if res:
                    queries.append(res)
                    print("Generated {} queries".format(len(queries)))
    file_path = os.path.join(config['data_dir'], dataset_name, "{}_new_udfs_labels.json".format(nunsupported_udfs))
    data = {"questions": queries}
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def generate_one_query(conn, ratio_lower_bound, ratio_upper_bound, npred, nattr_pred, nvars, depth, max_duration, nunsupported_udfs, supported_list, supported_attr_list, unsupported_list, unsupported_attr_list, dataset_name):
    """
    Generate one query with (npred) predicates and (nvars) variables.
    """
    # randomly choose nunsupported_udfs from unsupported_list
    # Example UDF format: {"predicate": "LeftOf", "parameter": None, "nargs": 2}

    # Randomly choose the number of supported/unsupported predicates/attribute predicates, while making sure that all of them are non-negative
    nunsupported_pred = random.randint(max(0, nunsupported_udfs - nattr_pred), min(nunsupported_udfs, npred))
    nunsupported_attr_pred = nunsupported_udfs - nunsupported_pred
    nsupported_pred = npred - nunsupported_pred
    nsupported_attr_pred = nattr_pred - nunsupported_attr_pred
    assert nunsupported_pred >= 0 and nunsupported_attr_pred >= 0 and nsupported_pred >= 0 and nsupported_attr_pred >= 0

    unsupported_predicates = random.sample(unsupported_list, nunsupported_udfs) + random.sample(unsupported_attr_list, nunsupported_attr_pred)
    for pred in unsupported_predicates:
        pred["variables"] = ["o_{}".format(i) for i in random.sample(list(range(nvars)), pred["nargs"])]
    candidate_predicate_list = []
    candidate_attr_predicate_list = []
    for (candidate_list, pred_list) in [(candidate_predicate_list, supported_list), (candidate_attr_predicate_list, supported_attr_list)]:
        for pred in pred_list:
            if pred["predicate"] in ["Near", "Far"]: # Order of variables doesn't matter
                combinations = itertools.combinations(["o_{}".format(i) for i in range(nvars)], pred["nargs"])
                for variables in combinations:
                    candidate_predicate = {"predicate": pred["predicate"], "parameter": pred["parameter"], "nargs": pred["nargs"], "variables": list(variables)}
                    candidate_list.append(candidate_predicate)
            else: # Order of variables matters
                permutations = itertools.permutations(["o_{}".format(i) for i in range(nvars)], pred["nargs"])
                for variables in permutations:
                    candidate_predicate = {"predicate": pred["predicate"], "parameter": pred["parameter"], "nargs": pred["nargs"], "variables": list(variables)}
                    candidate_list.append(candidate_predicate)
    supported_predicates = random.sample(candidate_predicate_list, nsupported_pred) + random.sample(candidate_attr_predicate_list, nsupported_attr_pred)
    predicate_list = unsupported_predicates + supported_predicates
    scene_graphs = [[] for _ in range(depth)]
    for pred in predicate_list:
        gid = random.randint(0, depth-1)
        scene_graphs[gid].append(pred)
    # remove empty scene graphs
    scene_graphs = [scene_graph for scene_graph in scene_graphs if len(scene_graph) > 0]
    # randomly choose duration for each scene graph, and make sure that at least one scene graph has duration > 1
    duration_unit = 5
    duration_values = [1]
    for i in range(1, max_duration // duration_unit + 1):
        duration_values.append(i * duration_unit)
    duration_per_scene_graph = [1 for _ in range(len(scene_graphs))]
    while sum(duration_per_scene_graph) == len(scene_graphs) and max_duration > 1:
        duration_per_scene_graph = [random.choice(duration_values) for _ in range(len(scene_graphs))]
    scene_graphs_str = []
    for i, scene_graph in enumerate(scene_graphs):
        scene_graph_str = print_scene_graph(scene_graph)
        if duration_per_scene_graph[i] > 1:
            scene_graph_str = "Duration({}, {})".format(scene_graph_str, duration_per_scene_graph[i])
        scene_graphs_str.append(scene_graph_str)
    query_str = "; ".join(scene_graphs_str)
    program = dsl_to_program(query_str)
    query_str = program_to_dsl(program, sort_variables=False) # Rewrite variables (but don't sort arguments within predicates)
    print("generated query", query_str)

    inputs_table_name = "Obj_clevrer"
    return generate_gt_labels_given_target_query(conn, query_str, ratio_lower_bound, ratio_upper_bound, dataset_name, inputs_table_name)


def generate_gt_labels_given_target_query(conn, query_str, ratio_lower_bound, ratio_upper_bound, dataset_name, inputs_table_name):
    """
    Given the target query (in string format), generate inputs.json and labels.json files containing
    inputs.json.

    ratio: minimum ratio of positive examples to negative examples
    """
    program = dsl_to_program(query_str)
    if inputs_table_name == "Obj_clevrer":
        input_vids = 10000
    else:
        raise ValueError("Unknown inputs_table_name: {}".format(inputs_table_name))
    _start = time.time()
    result, new_memo = duckdb_execute_clevrer_cache_sequence(conn, program, memo, inputs_table_name, input_vids)
    print("Time to execute query: {}".format(time.time() - _start))
    result = sorted(result)
    # lock.acquire()
    # for i, memo_dict in enumerate(new_memo):
    #     for k, v in memo_dict.items():
    #         memo[i][k] = v
    # lock.release()
    labels = []
    for i in range(input_vids):
        if i in result:
            labels.append(1)
        else:
            labels.append(0)

    print("Generated {} positive inputs and {} negative inputs".format(len(result), input_vids - len(result)))
    if len(result) / input_vids < ratio_lower_bound:
        print("Query {} doesn't have enough positive examples".format(query_str))
        return None
    if len(result) / input_vids > ratio_upper_bound:
        print("Query {} doesn't have enough negative examples".format(query_str))
        return None

    question_dict = dict(
        dsl=query_str,
        positive_videos=result,
        npos = sum(labels),
        nneg = len(labels) - sum(labels),
    )
    return question_dict


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_queries", type=int, default=10, help="number of queries to generate")
    ap.add_argument("--ratio_lower_bound", type=float, default=0.1, help="minimum ratio of positive examples to negative examples")
    ap.add_argument("--ratio_upper_bound", type=float, default=0.9, help="maximum ratio of positive examples to negative examples")
    ap.add_argument("--npred", type=int, default=5, help="number of predicates in each query")
    ap.add_argument("--nattr_pred", type=int, default=2, help="number of attribute predicates in each query")
    ap.add_argument("--nvars", type=int, default=3, help="number of variables in each query")
    ap.add_argument("--depth", type=int, default=3, help="number of region graphs in each query")
    ap.add_argument("--max_duration", type=int, default=15, help="maximum duration of each region graph")
    ap.add_argument("--nunsupported_udfs", type=int, default=0, help="number of unsupported UDFs in each query")
    ap.add_argument("--max_workers", type=int, default=10, help="number of workers")
    ap.add_argument("--dataset_name", type=str, default="clevrer", help="dataset name")

    args = ap.parse_args()
    n_queries = args.n_queries
    ratio_lower_bound = args.ratio_lower_bound
    ratio_upper_bound = args.ratio_upper_bound
    npred = args.npred
    nattr_pred = args.nattr_pred
    nvars = args.nvars
    depth = args.depth
    max_duration = args.max_duration
    nunsupported_udfs = args.nunsupported_udfs
    max_workers = args.max_workers
    dataset_name = args.dataset_name

    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

    supported_list = [
        {"predicate": "LeftOf", "parameter": None, "nargs": 2},
        {"predicate": "FrontOf", "parameter": None, "nargs": 2},
        {"predicate": "Left", "parameter": None, "nargs": 1},
        {"predicate": "Top", "parameter": None, "nargs": 1},
    ]
    supported_attr_list = [
        {"predicate": "Color", "parameter": "gray", "nargs": 1},
        {"predicate": "Color", "parameter": "red", "nargs": 1},
        {"predicate": "Color", "parameter": "blue", "nargs": 1},
        {"predicate": "Color", "parameter": "green", "nargs": 1},
        {"predicate": "Shape", "parameter": "cube", "nargs": 1},
        {"predicate": "Shape", "parameter": "sphere", "nargs": 1},
        {"predicate": "Material", "parameter": "rubber", "nargs": 1},
    ]
    unsupported_list = [
        {"predicate": "Near", "parameter": 1, "nargs": 2},
        {"predicate": "Far", "parameter": 3, "nargs": 2},
        {"predicate": "Behind", "parameter": None, "nargs": 2},
        {"predicate": "RightOf", "parameter": None, "nargs": 2},
        {"predicate": "Right", "parameter": None, "nargs": 1},
        {"predicate": "Bottom", "parameter": None, "nargs": 1},
    ]
    unsupported_attr_list = [
        {"predicate": "Color", "parameter": "brown", "nargs": 1},
        {"predicate": "Color", "parameter": "cyan", "nargs": 1},
        {"predicate": "Color", "parameter": "purple", "nargs": 1},
        {"predicate": "Color", "parameter": "yellow", "nargs": 1},
        {"predicate": "Shape", "parameter": "cylinder", "nargs": 1},
        {"predicate": "Material", "parameter": "metal", "nargs": 1},
    ]

    generate_clevrer_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nattr_pred, nvars, depth, max_duration, nunsupported_udfs, supported_list, supported_attr_list, unsupported_list, unsupported_attr_list, max_workers, dataset_name, config)