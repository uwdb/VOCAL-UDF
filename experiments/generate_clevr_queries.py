import json
import random
import itertools
import shutil
import numpy as np
import os
from src.utils import program_to_dsl, dsl_to_program, postgres_execute, postgres_execute_cache_sequence, print_scene_graph_helper
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
from vocaludf.utils import duckdb_execute_cache_sequence
from duckdb_dir.udf import register_udf
import yaml

random.seed(1234)
# np.random.seed(10)

m = multiprocessing.Manager()
lock = m.Lock()
memo = [LRU(10000) for _ in range(72159)]

def generate_clevr_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nvars, nunsupported_udfs, supported_list, unsupported_list, max_workers, dataset_name, config):
    """
    Generate (n_queries) queries with (npred) predicates and (nvars) variables. Each predicate is randomly chosen from (supported_list) and (nunsupported_udfs) unsupported UDFs from(unsupported_list), where each unsupported UDF must be used at least once.
    """

    conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=True)
    register_udf(conn)
    queries = []
    while len(queries) < n_queries:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(generate_one_query, repeat(conn, max_workers), repeat(ratio_lower_bound, max_workers), repeat(ratio_upper_bound, max_workers), repeat(npred, max_workers), repeat(nvars, max_workers), repeat(nunsupported_udfs, max_workers), repeat(supported_list, max_workers), repeat(unsupported_list, max_workers), repeat(dataset_name, max_workers)):
                if res:
                    queries.append(res)
                    print("Generated {} queries".format(len(queries)))
    file_path = os.path.join(config['data_dir'], "clevr", "{}_new_udfs_labels.json".format(nunsupported_udfs))
    data = {"questions": queries}
    # if os.path.exists(file_path):
    #     with open(file_path, "r") as file:
    #         data = json.load(file)
    # else:
    #     data = {"questions": []}
    # data["questions"].extend(queries)
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def generate_one_query(conn, ratio_lower_bound, ratio_upper_bound, npred, nvars, nunsupported_udfs, supported_list, unsupported_list, dataset_name):
    """
    Generate one query with (npred) predicates and (nvars) variables.
    """
    # randomly choose nunsupported_udfs from unsupported_list
    # Example UDF format: {"predicate": "LeftOf", "parameter": None, "nargs": 2}
    unsupported_predicates = random.sample(unsupported_list, nunsupported_udfs)
    for pred in unsupported_predicates:
        pred["variables"] = ["o_{}".format(i) for i in random.sample(list(range(nvars)), pred["nargs"])]
    candidate_predicate_list = []
    for pred in supported_list:
        if pred["predicate"] in ["EqualSize", "EqualMaterial", "EqualShape", "EqualColor"]:
            combinations = itertools.combinations(["o_{}".format(i) for i in range(nvars)], pred["nargs"])
            for variables in combinations:
                candidate_predicate = {"predicate": pred["predicate"], "parameter": pred["parameter"], "nargs": pred["nargs"], "variables": list(variables)}
                candidate_predicate_list.append(candidate_predicate)
        else:
            permutations = itertools.permutations(["o_{}".format(i) for i in range(nvars)], pred["nargs"])
            for variables in permutations:
                candidate_predicate = {"predicate": pred["predicate"], "parameter": pred["parameter"], "nargs": pred["nargs"], "variables": list(variables)}
                candidate_predicate_list.append(candidate_predicate)
    supported_predicates = random.sample(candidate_predicate_list, npred - nunsupported_udfs)
    predicate_list = unsupported_predicates + supported_predicates
    query_str = print_scene_graph_helper(predicate_list)
    program = dsl_to_program(query_str)
    query_str = program_to_dsl(program, sort_variables=False) # Rewrite variables (but don't sort arguments within predicates)
    print("generated query", query_str)

    inputs_table_name = "Obj_clevr"
    return generate_gt_labels_given_target_query(conn, query_str, ratio_lower_bound, ratio_upper_bound, dataset_name, inputs_table_name)


def generate_gt_labels_given_target_query(conn, query_str, ratio_lower_bound, ratio_upper_bound, dataset_name, inputs_table_name):
    """
    Given the target query (in string format), generate inputs.json and labels.json files containing
    inputs.json.

    ratio: minimum ratio of positive examples to negative examples
    """
    program = dsl_to_program(query_str)
    if inputs_table_name == "Obj_clevr":
        input_vids = 15000
    else:
        raise ValueError("Unknown inputs_table_name: {}".format(inputs_table_name))
    _start = time.time()
    result, new_memo = duckdb_execute_cache_sequence(conn, program, memo, inputs_table_name, input_vids)
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
        positive_images=result,
        npos = sum(labels),
        nneg = len(labels) - sum(labels),
    )
    return question_dict


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_queries", type=int, default=10, help="number of queries to generate")
    ap.add_argument("--ratio_lower_bound", type=float, default=0.1, help="minimum ratio of positive examples to negative examples")
    ap.add_argument("--ratio_upper_bound", type=float, default=0.9, help="maximum ratio of positive examples to negative examples")
    ap.add_argument("--npred", type=int, default=7, help="number of predicates in each query")
    ap.add_argument("--nvars", type=int, default=3, help="number of variables in each query")
    ap.add_argument("--nunsupported_udfs", type=int, default=0, help="number of unsupported UDFs in each query")
    ap.add_argument("--max_workers", type=int, default=10, help="number of workers")
    ap.add_argument("--dataset_name", type=str, default="clevr", help="dataset name")

    args = ap.parse_args()
    n_queries = args.n_queries
    ratio_lower_bound = args.ratio_lower_bound
    ratio_upper_bound = args.ratio_upper_bound
    npred = args.npred
    nvars = args.nvars
    nunsupported_udfs = args.nunsupported_udfs
    max_workers = args.max_workers
    dataset_name = args.dataset_name

    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

    supported_list = [
        {"predicate": "LeftOf", "parameter": None, "nargs": 2},
        {"predicate": "FrontOf", "parameter": None, "nargs": 2},
        {"predicate": "EqualSize", "parameter": None, "nargs": 2},
        {"predicate": "EqualMaterial", "parameter": None, "nargs": 2},
        {"predicate": "Size", "parameter": "big", "nargs": 1},
        {"predicate": "Color", "parameter": "gray", "nargs": 1},
        {"predicate": "Color", "parameter": "red", "nargs": 1},
        {"predicate": "Color", "parameter": "blue", "nargs": 1},
        {"predicate": "Color", "parameter": "green", "nargs": 1},
        {"predicate": "Shape", "parameter": "cube", "nargs": 1},
        {"predicate": "Shape", "parameter": "sphere", "nargs": 1},
        {"predicate": "Material", "parameter": "rubber", "nargs": 1},
    ]
    unsupported_list = [
        {"predicate": "Behind", "parameter": None, "nargs": 2},
        {"predicate": "RightOf", "parameter": None, "nargs": 2},
        {"predicate": "EqualShape", "parameter": None, "nargs": 2},
        {"predicate": "EqualColor", "parameter": None, "nargs": 2},
        {"predicate": "Size", "parameter": "small", "nargs": 1},
        {"predicate": "Color", "parameter": "brown", "nargs": 1},
        {"predicate": "Color", "parameter": "cyan", "nargs": 1},
        {"predicate": "Color", "parameter": "purple", "nargs": 1},
        {"predicate": "Color", "parameter": "yellow", "nargs": 1},
        {"predicate": "Shape", "parameter": "cylinder", "nargs": 1},
        {"predicate": "Material", "parameter": "metal", "nargs": 1},
    ]

    generate_clevr_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nvars, nunsupported_udfs, supported_list, unsupported_list, max_workers, dataset_name, config)