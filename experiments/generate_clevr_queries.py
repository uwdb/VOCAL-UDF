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
from vocaludf.utils import duckdb_execute_cache_sequence, replace_slot
from duckdb_dir.udf import register_udf
import yaml
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

random.seed(1234)
# np.random.seed(10)

m = multiprocessing.Manager()
lock = m.Lock()
memo = [LRU(10000) for _ in range(72159)]

config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

def generate_clevr_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nvars, nunsupported_udfs, supported_list, unsupported_list, max_workers, dataset_name):
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
    queries = queries[:n_queries]
    file_path = os.path.join(config['data_dir'], dataset_name, "{}_new_udfs_labels.json".format(nunsupported_udfs))
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

def generate_clevr_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nvars, udf_list, udf_must_include, max_workers, dataset_name):
    """
    Generate (n_queries) queries with (npred) predicates and (nvars) variables. Each predicate is randomly chosen from udf_list, and udf_must_include must appear at least once in the query.
    """
    conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=True)
    register_udf(conn)
    queries = []
    while len(queries) < n_queries:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(generate_one_query, repeat(conn, max_workers), repeat(ratio_lower_bound, max_workers), repeat(ratio_upper_bound, max_workers), repeat(npred, max_workers), repeat(nvars, max_workers), repeat(udf_list, max_workers), repeat(udf_must_include, max_workers), repeat(dataset_name, max_workers)):
                if res:
                    queries.append(res)
                    print("Generated {} queries".format(len(queries)))
    queries = queries[:n_queries]
    file_path = os.path.join(config['data_dir'], dataset_name, "queries_with_{}_labels.json".format(udf_must_include))
    data = {"questions": queries}
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def generate_one_query(conn, ratio_lower_bound, ratio_upper_bound, npred, nvars, udf_list, udf_must_include, dataset_name):
    """
    Generate one query with (npred) predicates and (nvars) variables.
    """
    def check_udf_must_include(udf_must_include, predicate_list):
        if '_' in udf_must_include:
            predicate, parameter = udf_must_include.split('_')
            for pred in predicate_list:
                if pred["predicate"] == predicate and pred["parameter"] == parameter:
                    return True
            return False
        else:
            for pred in predicate_list:
                if pred["predicate"] == udf_must_include:
                    return True
            return False

    predicate_list = []
    while not check_udf_must_include(udf_must_include, predicate_list):
        candidate_predicate_list = []
        for pred in udf_list:
            if pred["predicate"] in ["EqualSize", "EqualMaterial", "EqualShape", "EqualColor"]: # Order of variables doesn't matter
                combinations = itertools.combinations(["o_{}".format(i) for i in range(nvars)], pred["nargs"])
                for variables in combinations:
                    candidate_predicate = {"predicate": pred["predicate"], "parameter": pred["parameter"], "nargs": pred["nargs"], "variables": list(variables)}
                    candidate_predicate_list.append(candidate_predicate)
            else: # Order of variables matters
                permutations = itertools.permutations(["o_{}".format(i) for i in range(nvars)], pred["nargs"])
                for variables in permutations:
                    candidate_predicate = {"predicate": pred["predicate"], "parameter": pred["parameter"], "nargs": pred["nargs"], "variables": list(variables)}
                    candidate_predicate_list.append(candidate_predicate)
        predicate_list = random.sample(candidate_predicate_list, npred)

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

    # Use LLM to generate question str from query_str
    # NOTE: this step requires manual examination after generation
    with open(os.path.join(config['prompt_dir'], 'clevr_dsl2nl.txt'), 'r') as file:
        prompt = file.read()

    prompt = replace_slot(
        prompt,
        {"question": query_str}
    )

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        top_p=1,
        max_tokens=512,
        seed=42
    )

    nl_question = response.choices[0].message.content.lstrip('\n').rstrip('\n').lstrip('NL:')

    question_dict = dict(
        question=nl_question,
        visprog_program="",
        dsl=query_str,
        new_modules=[],
        positive_images=result,
        npos = sum(labels),
        nneg = len(labels) - sum(labels),
    )
    return question_dict

def generate_clevr_queries_with_unsupported_udfs():
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
        {"predicate": "Size_big", "parameter": None, "nargs": 1},
        {"predicate": "Color_gray", "parameter": None, "nargs": 1},
        {"predicate": "Color_red", "parameter": None, "nargs": 1},
        {"predicate": "Color_blue", "parameter": None, "nargs": 1},
        {"predicate": "Color_green", "parameter": None, "nargs": 1},
        {"predicate": "Shape_cube", "parameter": None, "nargs": 1},
        {"predicate": "Shape_sphere", "parameter": None, "nargs": 1},
        {"predicate": "Material_rubber", "parameter": None, "nargs": 1},
    ]
    unsupported_list = [
        {"predicate": "Behind", "parameter": None, "nargs": 2},
        {"predicate": "RightOf", "parameter": None, "nargs": 2},
        {"predicate": "EqualShape", "parameter": None, "nargs": 2},
        {"predicate": "EqualColor", "parameter": None, "nargs": 2},
        {"predicate": "Size_small", "parameter": None, "nargs": 1},
        {"predicate": "Color_brown", "parameter": None, "nargs": 1},
        {"predicate": "Color_cyan", "parameter": None, "nargs": 1},
        {"predicate": "Color_purple", "parameter": None, "nargs": 1},
        {"predicate": "Color_yellow", "parameter": None, "nargs": 1},
        {"predicate": "Shape_cylinder", "parameter": None, "nargs": 1},
        {"predicate": "Material_metal", "parameter": None, "nargs": 1},
    ]

    generate_clevr_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nvars, nunsupported_udfs, supported_list, unsupported_list, max_workers, dataset_name, config)

def generate_clevr_queries_udf_exclusion():
    # python generate_clevr_queries.py --udf "EqualShape"
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_queries", type=int, default=10, help="number of queries to generate")
    ap.add_argument("--ratio_lower_bound", type=float, default=0.1, help="minimum ratio of positive examples to negative examples")
    ap.add_argument("--ratio_upper_bound", type=float, default=0.9, help="maximum ratio of positive examples to negative examples")
    ap.add_argument("--npred", type=int, default=7, help="number of predicates in each query")
    ap.add_argument("--nvars", type=int, default=3, help="number of variables in each query")
    ap.add_argument("--max_workers", type=int, default=10, help="number of workers")
    ap.add_argument("--dataset_name", type=str, default="clevr", help="dataset name")
    ap.add_argument("--udf", type=str, help="UDF name that must be used in each query")

    args = ap.parse_args()
    n_queries = args.n_queries
    ratio_lower_bound = args.ratio_lower_bound
    ratio_upper_bound = args.ratio_upper_bound
    npred = args.npred
    nvars = args.nvars
    max_workers = args.max_workers
    dataset_name = args.dataset_name
    udf_must_include = args.udf

    udf_list = [
        {"predicate": "LeftOf", "parameter": None, "nargs": 2},
        {"predicate": "FrontOf", "parameter": None, "nargs": 2},
        {"predicate": "EqualSize", "parameter": None, "nargs": 2},
        {"predicate": "EqualMaterial", "parameter": None, "nargs": 2},
        {"predicate": "Size_big", "parameter": None, "nargs": 1},
        {"predicate": "Color_gray", "parameter": None, "nargs": 1},
        {"predicate": "Color_red", "parameter": None, "nargs": 1},
        {"predicate": "Color_blue", "parameter": None, "nargs": 1},
        {"predicate": "Color_green", "parameter": None, "nargs": 1},
        {"predicate": "Shape_cube", "parameter": None, "nargs": 1},
        {"predicate": "Shape_sphere", "parameter": None, "nargs": 1},
        {"predicate": "Material_rubber", "parameter": None, "nargs": 1},
        {"predicate": "Behind", "parameter": None, "nargs": 2},
        {"predicate": "RightOf", "parameter": None, "nargs": 2},
        {"predicate": "EqualShape", "parameter": None, "nargs": 2},
        {"predicate": "EqualColor", "parameter": None, "nargs": 2},
        {"predicate": "Size_small", "parameter": None, "nargs": 1},
        {"predicate": "Color_brown", "parameter": None, "nargs": 1},
        {"predicate": "Color_cyan", "parameter": None, "nargs": 1},
        {"predicate": "Color_purple", "parameter": None, "nargs": 1},
        {"predicate": "Color_yellow", "parameter": None, "nargs": 1},
        {"predicate": "Shape_cylinder", "parameter": None, "nargs": 1},
        {"predicate": "Material_metal", "parameter": None, "nargs": 1},
    ]

    generate_clevr_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nvars, udf_list, udf_must_include, max_workers, dataset_name)

if __name__ == '__main__':
    # generate_clevr_queries_with_unsupported_udfs()
    generate_clevr_queries_udf_exclusion()