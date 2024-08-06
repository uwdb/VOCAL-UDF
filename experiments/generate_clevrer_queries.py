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
from vocaludf.utils import duckdb_execute_cache_sequence, duckdb_execute_clevrer_cache_sequence, replace_slot, duckdb_execute_video_materialize
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

def generate_clevrer_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nattr_pred, nvars, depth, max_duration, nunsupported_udfs, supported_list, supported_attr_list, unsupported_list, unsupported_attr_list, max_workers, dataset_name):
    """
    Generate (n_queries) queries with (npred) predicates and (nvars) variables. Each predicate is randomly chosen from (supported_list) and (nunsupported_udfs) unsupported UDFs from(unsupported_list), where each unsupported UDF must be used at least once.
    """

    conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=True)

    init_table(conn, dataset_name)

    queries = []
    while len(queries) < n_queries:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(generate_one_query, repeat(conn, max_workers), repeat(ratio_lower_bound, max_workers), repeat(ratio_upper_bound, max_workers), repeat(npred, max_workers), repeat(nattr_pred, max_workers), repeat(nvars, max_workers), repeat(depth, max_workers), repeat(max_duration, max_workers), repeat(nunsupported_udfs, max_workers), repeat(supported_list, max_workers), repeat(supported_attr_list, max_workers), repeat(unsupported_list, max_workers), repeat(unsupported_attr_list, max_workers), repeat(dataset_name, max_workers)):
                if res:
                    queries.append(res)
                print("Generated {} queries".format(len(queries)))
    queries = queries[:n_queries]
    # file_path = os.path.join(config['data_dir'], dataset_name, "{}_new_udfs_labels.json".format(nunsupported_udfs))
    file_path = os.path.join(config['data_dir'], dataset_name, "unavailable_pred={}-npred={}-nattr_pred={}-nvars={}-depth={}-max_duration={}-lower_bound={}-upper_bound={}.json".format(nunsupported_udfs, npred, nattr_pred, nvars, depth, max_duration, ratio_lower_bound, ratio_upper_bound))
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

    unsupported_predicates = random.sample(unsupported_list, nunsupported_pred) + random.sample(unsupported_attr_list, nunsupported_attr_pred)
    for pred in unsupported_predicates:
        pred["variables"] = ["o_{}".format(i) for i in random.sample(list(range(nvars)), pred["nargs"])]
    udf_to_module = {
        "color_gray": "GRAY",
        "color_red": "RED",
        "color_blue": "BLUE",
        "color_green": "GREEN",
        "shape_cube": "CUBE",
        "shape_sphere": "SPHERE",
        "material_rubber": "RUBBER",
        "color_brown": "BROWN",
        "color_cyan": "CYAN",
        "color_purple": "PURPLE",
        "color_yellow": "YELLOW",
        "shape_cylinder": "CYLINDER",
        "material_metal": "METAL",
        "location_left": "LEFT",
        "location_top": "TOP",
        "location_right": "RIGHT",
        "location_bottom": "BOTTOM",
        "left_of": "LEFTOF",
        "front_of": "FRONTOF",
        "near": "NEAR",
        "far": "FAR",
        "behind": "BEHIND",
        "right_of": "RIGHTOF",
    }
    new_modules = [udf_to_module[pred["predicate"]] for pred in unsupported_predicates]
    random.shuffle(new_modules)

    candidate_predicate_list = []
    candidate_attr_predicate_list = []
    for (candidate_list, pred_list) in [(candidate_predicate_list, supported_list), (candidate_attr_predicate_list, supported_attr_list)]:
        for pred in pred_list:
            if pred["predicate"] in ["near", "far"]: # Order of variables doesn't matter
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

    return generate_gt_labels_given_target_query(conn, query_str, ratio_lower_bound, ratio_upper_bound, new_modules)

# def generate_clevrer_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nattr_pred, nvars, depth, max_duration, predicate_list, attr_predicate_list, udf_must_include, max_workers, dataset_name):
#     """
#     Generate (n_queries) queries with (npred) predicates and (nvars) variables. Each predicate is randomly chosen from predicate_list and attr_predicate_list, and udf_must_include must appear at least once in the query.
#     """

#     conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=True)
#     register_udf(conn)
#     queries = []
#     while len(queries) < n_queries:
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             for res in executor.map(generate_one_query, repeat(conn, max_workers), repeat(ratio_lower_bound, max_workers), repeat(ratio_upper_bound, max_workers), repeat(npred, max_workers), repeat(nattr_pred, max_workers), repeat(nvars, max_workers), repeat(depth, max_workers), repeat(max_duration, max_workers), repeat(predicate_list, max_workers), repeat(attr_predicate_list, max_workers), repeat(udf_must_include, max_workers), repeat(dataset_name, max_workers)):
#                 if res:
#                     queries.append(res)
#                     print("Generated {} queries".format(len(queries)))
#     queries = queries[:n_queries]
#     file_path = os.path.join(config['data_dir'], dataset_name, "queries_with_{}_labels.json".format(udf_must_include))
#     data = {"questions": queries}
#     with open(file_path, "w") as file:
#         json.dump(data, file, indent=4)


# def generate_one_query(conn, ratio_lower_bound, ratio_upper_bound, npred, nattr_pred, nvars, depth, max_duration, predicate_list, attr_predicate_list, udf_must_include, dataset_name):
#     """
#     Generate one query with (npred) predicates and (nvars) variables.
#     """
#     def check_udf_must_include(udf_must_include, selected_predicates):
#         # if '_' in udf_must_include:
#         #     predicate, parameter = udf_must_include.split('_')
#         #     for pred in selected_predicates:
#         #         if pred["predicate"] == predicate and pred["parameter"] == parameter:
#         #             return True
#         #     return False
#         # else:
#         for pred in selected_predicates:
#             if pred["predicate"] == udf_must_include:
#                 return True
#         return False

#     selected_predicates = []
#     while not check_udf_must_include(udf_must_include, selected_predicates):
#         candidate_predicate_list = []
#         candidate_attr_predicate_list = []
#         for (candidate_list, pred_list) in [(candidate_predicate_list, predicate_list), (candidate_attr_predicate_list, attr_predicate_list)]:
#             for pred in pred_list:
#                 if pred["predicate"] in ["Near", "Far"]: # Order of variables doesn't matter
#                     combinations = itertools.combinations(["o_{}".format(i) for i in range(nvars)], pred["nargs"])
#                     for variables in combinations:
#                         candidate_predicate = {"predicate": pred["predicate"], "parameter": pred["parameter"], "nargs": pred["nargs"], "variables": list(variables)}
#                         candidate_list.append(candidate_predicate)
#                 else: # Order of variables matters
#                     permutations = itertools.permutations(["o_{}".format(i) for i in range(nvars)], pred["nargs"])
#                     for variables in permutations:
#                         candidate_predicate = {"predicate": pred["predicate"], "parameter": pred["parameter"], "nargs": pred["nargs"], "variables": list(variables)}
#                         candidate_list.append(candidate_predicate)
#         selected_predicates = random.sample(candidate_predicate_list, npred) + random.sample(candidate_attr_predicate_list, nattr_pred)
#     scene_graphs = [[] for _ in range(depth)]
#     for pred in selected_predicates:
#         gid = random.randint(0, depth-1)
#         scene_graphs[gid].append(pred)
#     # remove empty scene graphs
#     scene_graphs = [scene_graph for scene_graph in scene_graphs if len(scene_graph) > 0]
#     # randomly choose duration for each scene graph, and make sure that at least one scene graph has duration > 1
#     duration_unit = 5
#     duration_values = [1]
#     for i in range(1, max_duration // duration_unit + 1):
#         duration_values.append(i * duration_unit)
#     duration_per_scene_graph = [1 for _ in range(len(scene_graphs))]
#     while sum(duration_per_scene_graph) == len(scene_graphs) and max_duration > 1:
#         duration_per_scene_graph = [random.choice(duration_values) for _ in range(len(scene_graphs))]
#     scene_graphs_str = []
#     for i, scene_graph in enumerate(scene_graphs):
#         scene_graph_str = print_scene_graph(scene_graph)
#         if duration_per_scene_graph[i] > 1:
#             scene_graph_str = "Duration({}, {})".format(scene_graph_str, duration_per_scene_graph[i])
#         scene_graphs_str.append(scene_graph_str)
#     query_str = "; ".join(scene_graphs_str)
#     program = dsl_to_program(query_str)
#     query_str = program_to_dsl(program, sort_variables=False) # Rewrite variables (but don't sort arguments within predicates)
#     print("generated query", query_str)

#     inputs_table_name = "Obj_clevrer"
#     return generate_gt_labels_given_target_query(conn, query_str, ratio_lower_bound, ratio_upper_bound, dataset_name, inputs_table_name)


def generate_gt_labels_given_target_query(conn, query_str, ratio_lower_bound, ratio_upper_bound, new_modules):
    """
    Given the target query (in string format), generate inputs.json and labels.json files containing
    inputs.json.

    ratio: minimum ratio of positive examples to negative examples
    """
    input_vids = 10000

    program = dsl_to_program(query_str)
    available_udf_names = ["color_gray", "color_red", "color_blue", "color_green", "shape_cube", "shape_sphere", "material_rubber", "color_brown", "color_cyan", "color_purple", "color_yellow", "shape_cylinder", "material_metal", "location_left", "location_top", "location_right", "location_bottom", "left_of", "front_of", "near", "far", "behind", "right_of"]

    _start = time.time()
    result = duckdb_execute_video_materialize(conn, program, None, available_udf_names, [], [])
    print("Time to execute query: {}".format(time.time() - _start))
    result = sorted(result)

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
    with open(os.path.join(config['prompt_dir'], 'clevrer_dsl2nl.txt'), 'r') as file:
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
        new_modules=new_modules,
        positive_videos=result,
        npos = sum(labels),
        nneg = len(labels) - sum(labels),
    )
    return question_dict


def init_table(conn, dataset):
    attribute_domain = ["color_gray", "color_red", "color_blue", "color_green", "shape_cube", "shape_sphere", "material_rubber", "color_brown", "color_cyan", "color_purple", "color_yellow", "shape_cylinder", "material_metal", "location_left", "location_top", "location_right", "location_bottom"]
    attr_parameters = ','.join('?' for _ in attribute_domain)
    sql = f"""
        CREATE TEMPORARY TABLE one_object AS
        SELECT
            o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
            o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
            320 AS height, 480 AS width
        FROM {dataset}_objects o
        LEFT OUTER JOIN {dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
        GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
    """
    print(f"Create one_object table:\n{sql}")
    conn.execute(sql, attribute_domain).df()

    relationship_domain = ["left_of", "front_of", "near", "far", "behind", "right_of"]
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
            320 AS height, 480 AS width
        FROM obj_with_attrs o1
        JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid
        LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
        LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
        WHERE o1.oid <> o2.oid
    """
    print(f"Create two_objects table:\n{sql}")
    conn.execute(sql, attribute_domain + relationship_domain).df()


def generate_clevrer_queries_with_unsupported_udfs():
    # python generate_clevrer_queries.py --n_queries 20 --npred 5 --nattr_pred 2 --nvars 3 --depth 3 --max_duration 15 --nunsupported_udfs 3 --dataset "clevrer" --ratio_lower_bound 0.05 --ratio_upper_bound 0.5
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_queries", type=int, default=10, help="number of queries to generate")
    ap.add_argument("--ratio_lower_bound", type=float, default=0.05, help="minimum ratio of positive examples to negative examples")
    ap.add_argument("--ratio_upper_bound", type=float, default=0.9, help="maximum ratio of positive examples to negative examples")
    ap.add_argument("--npred", type=int, default=5, help="number of predicates in each query")
    ap.add_argument("--nattr_pred", type=int, default=2, help="number of attribute predicates in each query")
    ap.add_argument("--nvars", type=int, default=3, help="number of variables in each query")
    ap.add_argument("--depth", type=int, default=3, help="number of region graphs in each query")
    ap.add_argument("--max_duration", type=int, default=15, help="maximum duration of each region graph")
    ap.add_argument("--nunsupported_udfs", type=int, default=3, help="number of unsupported UDFs in each query")
    ap.add_argument("--max_workers", type=int, default=1, help="number of workers")
    ap.add_argument("--dataset", type=str, default="clevrer", help="dataset name")

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
    dataset = args.dataset

    supported_list = [
        {"predicate": "left_of", "parameter": None, "nargs": 2},
        {"predicate": "front_of", "parameter": None, "nargs": 2},
        {"predicate": "location_left", "parameter": None, "nargs": 1},
        {"predicate": "location_top", "parameter": None, "nargs": 1},
    ]
    supported_attr_list = [
        {"predicate": "color_gray", "parameter": None, "nargs": 1},
        {"predicate": "color_red", "parameter": None, "nargs": 1},
        {"predicate": "color_blue", "parameter": None, "nargs": 1},
        {"predicate": "color_green", "parameter": None, "nargs": 1},
        {"predicate": "shape_cube", "parameter": None, "nargs": 1},
        {"predicate": "shape_sphere", "parameter": None, "nargs": 1},
        {"predicate": "material_rubber", "parameter": None, "nargs": 1},
    ]
    unsupported_list = [
        {"predicate": "near", "parameter": None, "nargs": 2},
        {"predicate": "far", "parameter": None, "nargs": 2},
        {"predicate": "behind", "parameter": None, "nargs": 2},
        {"predicate": "right_of", "parameter": None, "nargs": 2},
        {"predicate": "location_right", "parameter": None, "nargs": 1},
        {"predicate": "location_bottom", "parameter": None, "nargs": 1},
    ]
    unsupported_attr_list = [
        {"predicate": "color_brown", "parameter": None, "nargs": 1},
        {"predicate": "color_cyan", "parameter": None, "nargs": 1},
        {"predicate": "color_purple", "parameter": None, "nargs": 1},
        {"predicate": "color_yellow", "parameter": None, "nargs": 1},
        {"predicate": "shape_cylinder", "parameter": None, "nargs": 1},
        {"predicate": "material_metal", "parameter": None, "nargs": 1},
    ]

    generate_clevrer_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nattr_pred, nvars, depth, max_duration, nunsupported_udfs, supported_list, supported_attr_list, unsupported_list, unsupported_attr_list, max_workers, dataset)

# def generate_clevrer_queries_udf_exclusion():
#     # python generate_clevrer_queries.py --udf "Near"
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--n_queries", type=int, default=10, help="number of queries to generate")
#     ap.add_argument("--ratio_lower_bound", type=float, default=0.05, help="minimum ratio of positive examples to negative examples")
#     ap.add_argument("--ratio_upper_bound", type=float, default=0.9, help="maximum ratio of positive examples to negative examples")
#     ap.add_argument("--npred", type=int, default=5, help="number of predicates in each query")
#     ap.add_argument("--nattr_pred", type=int, default=2, help="number of attribute predicates in each query")
#     ap.add_argument("--nvars", type=int, default=3, help="number of variables in each query")
#     ap.add_argument("--depth", type=int, default=3, help="number of region graphs in each query")
#     ap.add_argument("--max_duration", type=int, default=15, help="maximum duration of each region graph")
#     ap.add_argument("--max_workers", type=int, default=10, help="number of workers")
#     ap.add_argument("--dataset_name", type=str, default="clevrer", help="dataset name")
#     ap.add_argument("--udf", type=str, help="UDF name that must be used in each query")

#     args = ap.parse_args()
#     n_queries = args.n_queries
#     ratio_lower_bound = args.ratio_lower_bound
#     ratio_upper_bound = args.ratio_upper_bound
#     npred = args.npred
#     nattr_pred = args.nattr_pred
#     nvars = args.nvars
#     depth = args.depth
#     max_duration = args.max_duration
#     max_workers = args.max_workers
#     dataset_name = args.dataset_name
#     udf_must_include = args.udf

#     predicate_list = [
#         {"predicate": "LeftOf", "parameter": None, "nargs": 2},
#         {"predicate": "FrontOf", "parameter": None, "nargs": 2},
#         {"predicate": "Left", "parameter": None, "nargs": 1},
#         {"predicate": "Top", "parameter": None, "nargs": 1},
#         {"predicate": "Near", "parameter": None, "nargs": 2},
#         {"predicate": "Far", "parameter": None, "nargs": 2},
#         {"predicate": "Behind", "parameter": None, "nargs": 2},
#         {"predicate": "RightOf", "parameter": None, "nargs": 2},
#         {"predicate": "Right", "parameter": None, "nargs": 1},
#         {"predicate": "Bottom", "parameter": None, "nargs": 1},
#     ]
#     attr_predicate_list = [
#         {"predicate": "Color_gray", "parameter": None, "nargs": 1},
#         {"predicate": "Color_red", "parameter": None, "nargs": 1},
#         {"predicate": "Color_blue", "parameter": None, "nargs": 1},
#         {"predicate": "Color_green", "parameter": None, "nargs": 1},
#         {"predicate": "Shape_cube", "parameter": None, "nargs": 1},
#         {"predicate": "Shape_sphere", "parameter": None, "nargs": 1},
#         {"predicate": "Material_rubber", "parameter": None, "nargs": 1},
#         {"predicate": "Color_brown", "parameter": None, "nargs": 1},
#         {"predicate": "Color_cyan", "parameter": None, "nargs": 1},
#         {"predicate": "Color_purple", "parameter": None, "nargs": 1},
#         {"predicate": "Color_yellow", "parameter": None, "nargs": 1},
#         {"predicate": "Shape_cylinder", "parameter": None, "nargs": 1},
#         {"predicate": "Material_metal", "parameter": None, "nargs": 1},
#     ]

#     generate_clevrer_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, nattr_pred, nvars, depth, max_duration, predicate_list, attr_predicate_list, udf_must_include, max_workers, dataset_name)

if __name__ == '__main__':
    generate_clevrer_queries_with_unsupported_udfs()
    # generate_clevrer_queries_udf_exclusion()