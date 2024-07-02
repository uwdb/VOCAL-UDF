from src.utils import program_to_dsl, dsl_to_program, postgres_execute, postgres_execute_cache_sequence, print_scene_graph_helper
import duckdb
import itertools
import copy
from vocaludf.parser import parse, parse_udf
import json
import logging
import os
from torch.utils.data import IterableDataset
import torch
import pandas as pd
import math
from typing import List
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def exception_hook(exc_type, exc_value, exc_traceback, logger=logger):
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )

# TODO: We can also support video loading and feature extraction on the fly, without creating the parquet files,
# since writing parquet files is the bottleneck right now.
class PredImageDataset(IterableDataset):
    def __init__(self, conn, n_obj, attribute_features_dir, relationship_features_dir):
        self.conn = conn
        self.n_obj = n_obj
        self.features_dir = attribute_features_dir if n_obj == 1 else relationship_features_dir

        self.files = self._get_files(self.features_dir)

        self.num_files = len(self.files)
        self.start = 0
        self.end = self.num_files
        self.len = duckdb.sql(f"SELECT COUNT(*) FROM '{self.features_dir}/*.parquet';").fetchall()[0][0]
        self.columns = ['vid', 'fid', 'o1_oid', 'feature'] if n_obj == 1 else ['vid', 'fid', 'o1_oid', 'o2_oid', 'feature']
    def _get_files(self, features_dir, extension=".parquet") -> List[str]:
        all_files = os.listdir(features_dir)
        matched_files = sorted([os.path.join(features_dir, f) for f in all_files if f.endswith(extension)])
        return matched_files

    def __len__(self):
        return self.len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else: # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        for file in self.files[iter_start:iter_end]:
            df = pd.read_parquet(file, columns=self.columns)

            feature_col = df["feature"].values
            metadata = df.drop('feature', axis=1)

            for (_, row), feature in zip(metadata.iterrows(), feature_col):
                yield np.array(row.to_numpy(dtype=int)), torch.tensor(feature, dtype=torch.float32)


def parse_signature(signature):
    """
    Example:
    signature: "Color_red(o1, -1)"
    parsed result: {'fn_name': 'Color_red', 'variables': ['o1'], 'parameter': -1}
    """
    # NOTE: could throw an exception if the signature is not in the correct format
    result = parse_udf().parseString(signature, parseAll=True).as_dict()
    udf_name = result["fn_name"]
    udf_vars = result["variables"]
    # tokens = list(tokenize.generate_tokens(io.StringIO(signature).readline))
    # udf_name = tokens[0].string
    # udf_vars = [token for token in tokens[2:-3] if token.string not in [',','=']]
    return udf_name, udf_vars


def transform_function(original_code, instantiation_dict):
    """
    Transforms the original code by removing **kwargs from the function definition
    and inserting a line defining kwargs with the corrected string format.

    Args:
        original_code (str): The original code to be transformed.
        instantiation_dict (dict): The dictionary containing the values for kwargs.

    Returns:
        str: The transformed code.
    """
    # Split the original function into lines
    lines = original_code.split('\n')

    # Find the line with the function definition and remove **kwargs
    for i, line in enumerate(lines):
        if line.startswith('def ') and '**kwargs' in line:
            # Replace **kwargs with nothing
            lines[i] = line.replace(', **kwargs', '').replace('**kwargs, ', '').replace('**kwargs', '')

            # Insert the line defining kwargs with the corrected string format
            kwargs_str = json.dumps(instantiation_dict)
            kwargs_line = f"    kwargs = {kwargs_str}"
            lines.insert(i + 1, kwargs_line)
            break

    # Rejoin the modified lines into a single string
    transformed_code = '\n'.join(lines)

    return transformed_code

def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key +"}}", value.replace('"', "'"))
    return text


def get_active_domain(config, dataset, registered_functions):
    object_domain = config[dataset]['onames']
    relationship_domain = []
    attribute_domain = []

    for registered_function in registered_functions:
        signature = registered_function["signature"]
        registered_function_name, registered_function_vars = parse_signature(signature)
        if registered_function_name.lower() == "object":
            continue
        if len(registered_function_vars) == 2:
            # Relationship UDF
            relationship_domain.append(registered_function_name.lower())
        else:
            # Attribute UDF
            attribute_domain.append(registered_function_name.lower())
    return object_domain, relationship_domain, attribute_domain


def duckdb_execute_cache_sequence(conn, current_query, memo, inputs_table_name, input_vids, table_as_input_to_udf=False):
    """
    This method uses temp views and only caches binary query predictions.
    input_vids: list of video segment ids. For image datasets, this is actually fids.
    Example query:
        Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, Behind)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    Only cache binary query prediction, rather than [vid, fid, oids]: cache[graph] = 1 (positive) or 0 (negative)
    Output:
    new_memoize: new cached results from this query, which will be added to the global cache (for multi-threading)
    """
    if inputs_table_name.startswith("Obj_warsaw") or inputs_table_name.startswith("Obj_shibuya"):
        is_traffic = True
    else:
        is_traffic = False

    with conn.cursor() as cur:
        new_memo = [{} for _ in range(len(memo))]

        output_vids = []
        # Prepare cache result
        if isinstance(input_vids, int):
            remaining_vids = set(range(input_vids))
        else:
            remaining_vids = set(input_vids)

        signatures = []
        for i in range(len(current_query)):
            seq_signature = program_to_dsl(current_query[:(i+1)], True)
            signatures.append(seq_signature)

        filtered_vids = []
        cached_output_vids = []
        for vid in remaining_vids:
            for i, seq_signature in enumerate(signatures):
                if seq_signature not in memo[vid]:
                    filtered_vids.append(vid)
                elif memo[vid][seq_signature] == 0:
                    break
                elif i == len(signatures) - 1: # The full query predicates it as positive
                    cached_output_vids.append(vid)

        # select input videos
        # cur.execute("CREATE INDEX IF NOT EXISTS idx_{t} ON {t} (fid);".format(t=inputs_table_name))
        parameters = ','.join('?' for _ in filtered_vids)
        cur.execute("CREATE TEMPORARY TABLE Obj_filtered ON COMMIT DROP AS SELECT * FROM {inputs_table_name} WHERE fid = ANY([{parameters}]);".format(inputs_table_name=inputs_table_name, parameters=parameters), filtered_vids)
        # cur.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (fid);")
        # print("select input videos: ", time.time() - _start)
        encountered_variables = []

        assert(len(current_query) == 1)
        dict = current_query[0]
        assert(dict["duration_constraint"] == 1)
        scene_graph = dict["scene_graph"]
        for p in scene_graph:
            for v in p["variables"]:
                if v not in encountered_variables:
                    encountered_variables.append(v)

        # Execute for unseen videos
        encountered_variables = sorted(encountered_variables, key=lambda x: int(x[1:]))
        tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables])
        where_clauses = []
        for i in range(len(encountered_variables)-1):
            where_clauses.append("{v1}.fid = {v2}.fid".format(v1=encountered_variables[i], v2=encountered_variables[i+1])) # join variables
        for p in scene_graph:
            predicate = p["predicate"]
            parameter = p.get("parameter", None)
            variables = p["variables"]
            if table_as_input_to_udf:
                args = ", ".join([v for v in variables])
            else:
                args = []
                for v in variables:
                    if is_traffic:
                        args.append("{v}.x1, {v}.y1, {v}.x2, {v}.y2, {v}.vx, {v}.vy, {v}.ax, {v}.ay".format(v=v))
                    else:
                        args.append("{v}.shape, {v}.color, {v}.material, {v}.x1, {v}.y1, {v}.x2, {v}.y2".format(v=v))
                args = ", ".join(args)
            if parameter:
                if isinstance(parameter, str):
                    args = "'{}', {}".format(parameter, args)
                else:
                    args = "{}, {}".format(parameter, args)
            where_clauses.append("{}({}) = true".format(predicate, args))
        # For general case
        for var_pair in itertools.combinations(encountered_variables, 2):
            where_clauses.append("{}.oid <> {}.oid".format(var_pair[0], var_pair[1]))
        where_clauses = " and ".join(where_clauses)
        sql_string = """SELECT DISTINCT {}.fid as fid FROM {} WHERE {};""".format(encountered_variables[0], tables, where_clauses)
        # print(sql_string)
        cur.execute(sql_string)
        res = cur.fetchall()
        output_vids = [row[0] for row in res]
        # # Store new cached results
        # for input_vid in filtered_vids:
        #     if input_vid in output_vids:
        #         new_memo[input_vid][signatures[graph_idx]] = 1
        #     else:
        #         new_memo[input_vid][signatures[graph_idx]] = 0
        output_vids.extend(cached_output_vids)

        # Commit
        conn.commit()
    return output_vids, new_memo


def duckdb_execute_clevrer_dataframe(conn, current_query, memo, inputs_table_name, input_vids):
    """
    This method uses temp views and only caches binary query predictions.
    input_vids: list of video segment ids. For image datasets, this is actually fids.
    Example query:
        Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, Behind)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    Only cache binary query prediction, rather than [vid, fid, oids]: cache[graph] = 1 (positive) or 0 (negative)
    Output:
    new_memoize: new cached results from this query, which will be added to the global cache (for multi-threading)
    """
    if inputs_table_name.startswith("Obj_warsaw") or inputs_table_name.startswith("Obj_shibuya"):
        is_traffic = True
    else:
        is_traffic = False

    with conn.cursor() as cur:
        new_memo = [{} for _ in range(len(memo))]

        output_vids = []
        # Prepare cache result
        if isinstance(input_vids, int):
            remaining_vids = set(range(input_vids))
        else:
            remaining_vids = set(input_vids)

        signatures = []
        for i in range(len(current_query)):
            seq_signature = program_to_dsl(current_query[:(i+1)], True)
            signatures.append(seq_signature)

        filtered_vids = []
        cached_output_vids = []
        for vid in remaining_vids:
            for i, seq_signature in enumerate(signatures):
                if seq_signature not in memo[vid]:
                    filtered_vids.append(vid)
                elif memo[vid][seq_signature] == 0:
                    break
                elif i == len(signatures) - 1: # The full query predicates it as positive
                    cached_output_vids.append(vid)

        # select input videos
        # cur.execute("CREATE INDEX IF NOT EXISTS idx_{t} ON {t} (fid);".format(t=inputs_table_name))
        parameters = ','.join('?' for _ in filtered_vids)
        # cur.execute("CREATE TEMPORARY VIEW Obj_filtered AS SELECT * FROM {inputs_table_name} WHERE vid < {input_vids};".format(inputs_table_name=inputs_table_name, input_vids=input_vids))
        Obj_filtered = cur.execute("SELECT * FROM {inputs_table_name} WHERE vid = ANY([{parameters}]);".format(inputs_table_name=inputs_table_name, parameters=parameters), filtered_vids).fetchdf()
        print("here1")
        # cur.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (fid);")
        # print("select input videos: ", time.time() - _start)
        encountered_variables_prev_graphs = []
        encountered_variables_current_graph = []
        print(current_query)
        for graph_idx, dict in enumerate(current_query):
            print(graph_idx)
            scene_graph = dict["scene_graph"]
            duration_constraint = dict["duration_constraint"]
            for p in scene_graph:
                for v in p["variables"]:
                    if v not in encountered_variables_current_graph:
                        encountered_variables_current_graph.append(v)

            # Execute for unseen videos
            encountered_variables_current_graph = sorted(encountered_variables_current_graph, key=lambda x: int(x[1:]))
            tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables_current_graph])
            where_clauses = []
            for i in range(len(encountered_variables_current_graph)-1):
                where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_current_graph[i], v2=encountered_variables_current_graph[i+1])) # join variables
            for p in scene_graph:
                predicate = p["predicate"]
                parameter = p.get("parameter", None)
                variables = p["variables"]
                args = []
                for v in variables:
                    if is_traffic:
                        args.append("{v}.x1, {v}.y1, {v}.x2, {v}.y2, {v}.vx, {v}.vy, {v}.ax, {v}.ay".format(v=v))
                    else:
                        args.append("{v}.shape, {v}.color, {v}.material, {v}.x1, {v}.y1, {v}.x2, {v}.y2".format(v=v))
                args = ", ".join(args)
                if parameter:
                    if isinstance(parameter, str):
                        args = "'{}', {}".format(parameter, args)
                    else:
                        args = "{}, {}".format(parameter, args)
                where_clauses.append("{}({}) = true".format(predicate, args))
            # For general case
            for var_pair in itertools.combinations(encountered_variables_current_graph, 2):
                where_clauses.append("{}.oid <> {}.oid".format(var_pair[0], var_pair[1]))
            where_clauses = " and ".join(where_clauses)
            fields = "{v}.vid as vid, {v}.fid as fid, ".format(v=encountered_variables_current_graph[0])
            fields += ", ".join(["{v}.oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
            oid_list = ["{}_oid".format(v) for v in encountered_variables_current_graph]
            oids = ", ".join(oid_list)
            sql_string = """SELECT {} FROM {} WHERE {};""".format(fields, tables, where_clauses)
            print(sql_string)
            graph_df = cur.execute(sql_string).fetchdf()
            print("here2")
            # print("execute for unseen videos: ", time.time() - _start_execute)
            # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

            if graph_idx > 0:
                obj_union = copy.deepcopy(encountered_variables_prev_graphs)
                obj_union_fields = []
                obj_intersection_fields = []
                for v in encountered_variables_prev_graphs:
                    obj_union_fields.append("t0.{}_oid".format(v))
                for v in encountered_variables_current_graph:
                    if v in encountered_variables_prev_graphs:
                        obj_intersection_fields.append("t0.{v}_oid = t1.{v}_oid".format(v=v))
                    else:
                        for u in encountered_variables_prev_graphs:
                            obj_intersection_fields.append("t0.{u}_oid <> t1.{v}_oid".format(u=u, v=v))
                        obj_union.append(v)
                        obj_union_fields.append("t1.{}_oid".format(v))
                obj_union_fields = ", ".join(obj_union_fields)
                obj_intersection_fields = " and ".join(obj_intersection_fields)
                sql_string = """
                SELECT t0.vid, t1.fid, {obj_union_fields}
                FROM graph_contiguous t0, graph_df t1
                WHERE t0.vid = t1.vid AND {obj_intersection_fields} AND t0.fid < t1.fid
                """.format(graph_idx_prev=graph_idx-1, obj_union_fields=obj_union_fields, obj_intersection_fields=obj_intersection_fields)
                # print(sql_string)
                graph_df = cur.execute(sql_string).fetchdf()
                print("here3")
                # temp_views.append("g{}_filtered".format(graph_idx))
            else:
                obj_union = encountered_variables_current_graph

            # Generate scene graph sequence:
            obj_union_fields = ", ".join(["{}_oid".format(v) for v in obj_union])
            sql_string = """
                SELECT vid, fid, {obj_union_fields},
                lead(fid, {duration_constraint} - 1, 0) OVER (PARTITION BY vid, {obj_union_fields} ORDER BY fid) as fid_offset
                FROM graph_df
            """.format(graph_idx=graph_idx, duration_constraint=duration_constraint, obj_union_fields=obj_union_fields)
            # print(sql_string)
            graph_windowed_df = cur.execute(sql_string).fetchdf()
            print("here4")
            # print("windowed: ", time.time() - _start_windowed)

            sql_string = """
                SELECT vid, {obj_union_fields}, min(fid_offset) AS fid
                FROM graph_windowed_df
                WHERE fid_offset = fid + ({duration_constraint} - 1)
                GROUP BY vid, {obj_union_fields}
            """.format(graph_idx=graph_idx, obj_union_fields=obj_union_fields, duration_constraint=duration_constraint)
            # print(sql_string)
            graph_contiguous = cur.execute(sql_string).fetchdf()
            print(graph_contiguous)
            print("here5")
            print("here6")
            cur.execute("SELECT DISTINCT vid FROM graph_contiguous".format(graph_idx))
            res = cur.fetchall()
            print("here6")
            output_vids = [row[0] for row in res]
            # Store new cached results
            for input_vid in filtered_vids:
                if input_vid in output_vids:
                    new_memo[input_vid][signatures[graph_idx]] = 1
                else:
                    new_memo[input_vid][signatures[graph_idx]] = 0
            print("here7")
            encountered_variables_prev_graphs = obj_union
            encountered_variables_current_graph = []
        output_vids.extend(cached_output_vids)

        # Drop views
        # cur.execute("DROP VIEW {}".format(", ".join(temp_views)))
        # Commit
        conn.commit()
        # print(output_vids)
    return output_vids, new_memo

def duckdb_execute_clevrer_cache_sequence(conn, current_query, memo, inputs_table_name, input_vids, table_as_input_to_udf=False):
    """
    This method uses temp views and only caches binary query predictions.
    input_vids: list of video segment ids. For image datasets, this is actually fids.
    Example query:
        Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, Behind)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    Only cache binary query prediction, rather than [vid, fid, oids]: cache[graph] = 1 (positive) or 0 (negative)
    Output:
    new_memoize: new cached results from this query, which will be added to the global cache (for multi-threading)
    """
    if inputs_table_name.startswith("Obj_warsaw") or inputs_table_name.startswith("Obj_shibuya"):
        is_traffic = True
    else:
        is_traffic = False

    temp_views = []

    with conn.cursor() as cur:
        new_memo = [{} for _ in range(len(memo))]

        output_vids = []
        # Prepare cache result
        if isinstance(input_vids, int):
            remaining_vids = set(range(input_vids))
        else:
            remaining_vids = set(input_vids)

        signatures = []
        for i in range(len(current_query)):
            seq_signature = program_to_dsl(current_query[:(i+1)], True)
            signatures.append(seq_signature)

        filtered_vids = []
        cached_output_vids = []
        for vid in remaining_vids:
            for i, seq_signature in enumerate(signatures):
                if seq_signature not in memo[vid]:
                    filtered_vids.append(vid)
                elif memo[vid][seq_signature] == 0:
                    break
                elif i == len(signatures) - 1: # The full query predicates it as positive
                    cached_output_vids.append(vid)

        # select input videos
        parameters = ','.join('?' for _ in filtered_vids)
        cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {inputs_table_name} WHERE vid < {input_vids};".format(inputs_table_name=inputs_table_name, input_vids=input_vids))
        # Obj_filtered = cur.execute("SELECT * FROM {inputs_table_name} WHERE vid = ANY([{parameters}]);".format(inputs_table_name=inputs_table_name, parameters=parameters), filtered_vids).fetchdf()
        # print("here1")
        # cur.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (vid, fid);")
        # print("select input videos: ", time.time() - _start)
        encountered_variables_prev_graphs = []
        encountered_variables_current_graph = []
        for graph_idx, dict in enumerate(current_query):
            # Generate scene graph:
            scene_graph = dict["scene_graph"]
            duration_constraint = dict["duration_constraint"]
            for p in scene_graph:
                for v in p["variables"]:
                    if v not in encountered_variables_current_graph:
                        encountered_variables_current_graph.append(v)

            # Execute for unseen videos
            encountered_variables_current_graph = sorted(encountered_variables_current_graph, key=lambda x: int(x[1:]))
            tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables_current_graph])
            where_clauses = []
            for i in range(len(encountered_variables_current_graph)-1):
                where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_current_graph[i], v2=encountered_variables_current_graph[i+1])) # join variables
            for p in scene_graph:
                predicate = p["predicate"]
                parameter = p.get("parameter", None)
                variables = p["variables"]
                if table_as_input_to_udf:
                    args = ", ".join([v for v in variables])
                else:
                    args = []
                    for v in variables:
                        if is_traffic:
                            args.append("{v}.x1, {v}.y1, {v}.x2, {v}.y2, {v}.vx, {v}.vy, {v}.ax, {v}.ay".format(v=v))
                        else:
                            args.append("{v}.shape, {v}.color, {v}.material, {v}.x1, {v}.y1, {v}.x2, {v}.y2".format(v=v))
                    args = ", ".join(args)
                if parameter:
                    if isinstance(parameter, str):
                        args = "'{}', {}".format(parameter, args)
                    else:
                        args = "{}, {}".format(parameter, args)
                where_clauses.append("{}({}) = true".format(predicate, args))
            # For general case
            for var_pair in itertools.combinations(encountered_variables_current_graph, 2):
                where_clauses.append("{}.oid <> {}.oid".format(var_pair[0], var_pair[1]))
            where_clauses = " and ".join(where_clauses)
            fields = "{v}.vid as vid, {v}.fid as fid, ".format(v=encountered_variables_current_graph[0])
            fields += ", ".join(["{v}.oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
            oid_list = ["{}_oid".format(v) for v in encountered_variables_current_graph]
            oids = ", ".join(oid_list)
            sql_string = """CREATE TEMPORARY TABLE g{} AS SELECT {} FROM {} WHERE {};""".format(graph_idx, fields, tables, where_clauses)
            # print(sql_string)
            cur.execute(sql_string)
            temp_views.append("g{}".format(graph_idx))
            # print("execute for unseen videos: ", time.time() - _start_execute)
            # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

            if graph_idx > 0:
                obj_union = copy.deepcopy(encountered_variables_prev_graphs)
                obj_union_fields = []
                obj_intersection_fields = []
                for v in encountered_variables_prev_graphs:
                    obj_union_fields.append("t0.{}_oid".format(v))
                for v in encountered_variables_current_graph:
                    if v in encountered_variables_prev_graphs:
                        obj_intersection_fields.append("t0.{v}_oid = t1.{v}_oid".format(v=v))
                    else:
                        for u in encountered_variables_prev_graphs:
                            obj_intersection_fields.append("t0.{u}_oid <> t1.{v}_oid".format(u=u, v=v))
                        obj_union.append(v)
                        obj_union_fields.append("t1.{}_oid".format(v))
                obj_union_fields = ", ".join(obj_union_fields)
                obj_intersection_fields = " and ".join(obj_intersection_fields)
                sql_string = """
                CREATE TEMPORARY TABLE g{graph_idx}_filtered AS (
                    SELECT t0.vid, t1.fid, {obj_union_fields}
                    FROM g{graph_idx_prev}_contiguous t0, g{graph_idx} t1
                    WHERE t0.vid = t1.vid AND {obj_intersection_fields} AND t0.fid < t1.fid
                );
                """.format(graph_idx=graph_idx, graph_idx_prev=graph_idx-1, obj_union_fields=obj_union_fields, obj_intersection_fields=obj_intersection_fields)
                # print(sql_string)
                cur.execute(sql_string)
                temp_views.append("g{}_filtered".format(graph_idx))
            else:
                obj_union = encountered_variables_current_graph

            # Generate scene graph sequence:
            table_name = "g{}_filtered".format(graph_idx) if graph_idx > 0 else "g{}".format(graph_idx)
            obj_union_fields = ", ".join(["{}_oid".format(v) for v in obj_union])
            sql_string = """
                CREATE TEMPORARY TABLE g{graph_idx}_windowed AS (
                SELECT vid, fid, {obj_union_fields},
                lead(fid, {duration_constraint} - 1, 0) OVER (PARTITION BY vid, {obj_union_fields} ORDER BY fid) as fid_offset
                FROM {table_name}
            );
            """.format(graph_idx=graph_idx, duration_constraint=duration_constraint, obj_union_fields=obj_union_fields, table_name=table_name)
            # print(sql_string)
            cur.execute(sql_string)
            temp_views.append("g{}_windowed".format(graph_idx))
            # print("windowed: ", time.time() - _start_windowed)

            sql_string = """
                CREATE TEMPORARY TABLE g{graph_idx}_contiguous AS (
                SELECT vid, {obj_union_fields}, min(fid_offset) AS fid
                FROM g{graph_idx}_windowed
                WHERE fid_offset = fid + ({duration_constraint} - 1)
                GROUP BY vid, {obj_union_fields}
            );
            """.format(graph_idx=graph_idx, obj_union_fields=obj_union_fields, duration_constraint=duration_constraint)
            # print(sql_string)
            cur.execute(sql_string)
            temp_views.append("g{}_contiguous".format(graph_idx))
            cur.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(graph_idx))
            res = cur.fetchall()
            output_vids = [row[0] for row in res]
            # Store new cached results
            for input_vid in filtered_vids:
                if input_vid in output_vids:
                    new_memo[input_vid][signatures[graph_idx]] = 1
                else:
                    new_memo[input_vid][signatures[graph_idx]] = 0
            encountered_variables_prev_graphs = obj_union
            encountered_variables_current_graph = []
        output_vids.extend(cached_output_vids)

        # Drop views
        # for temp_view in temp_views:
        #     cur.execute("DROP VIEW {}".format(temp_view))
        # cur.execute("DROP VIEW {}".format(", ".join(temp_views)))
        # Commit
        conn.commit()
    return output_vids, new_memo


def duckdb_execute_video_materialize(conn, current_query, input_vids, available_udf_names, materialized_udf_names, on_the_fly_udf_names):
    """
    There are three types of UDFs:
    - available_udf_names: available UDFs
    - materialized_udf_names: new UDFs, with results already materialized
    - on_the_fly_udf_names: new UDFs, with results computed on-the-fly
    """
    # Duration((LeftOf(o2, o0), Color_Red(o0), FrontOf(o1, o2)), 15); Duration((FarFrom(o1, o2), LeftOf(o0, o2)), 5); (Behind(o2, o0), Material_Metal(o1))

    output_vids = []

    temp_tables = []
    # select input videos
    if input_vids is None:
        conn.execute(f"CREATE TEMPORARY TABLE obj_attr_filtered AS SELECT * FROM one_object;")
        conn.execute(f"CREATE TEMPORARY TABLE rel_filtered AS SELECT * FROM two_objects;")
    elif isinstance(input_vids, int):
        conn.execute(f"CREATE TEMPORARY TABLE obj_attr_filtered AS SELECT * FROM one_object WHERE vid < {input_vids};")
        conn.execute(f"CREATE TEMPORARY TABLE rel_filtered AS SELECT * FROM two_objects WHERE vid < {input_vids};")
    else: # list of input_vids
        parameters = ','.join('?' for _ in input_vids)
        conn.execute(f"CREATE TEMPORARY TABLE obj_attr_filtered AS SELECT * FROM one_object WHERE vid = ANY([{parameters}]);", input_vids)
        conn.execute(f"CREATE TEMPORARY TABLE rel_filtered AS SELECT * FROM two_objects WHERE vid = ANY([{parameters}]);", input_vids)
    temp_tables.append("obj_attr_filtered")
    temp_tables.append("rel_filtered")
    # Obj_filtered = conn.execute("SELECT * FROM {inputs_table_name} WHERE vid = ANY([{parameters}]);".format(inputs_table_name=inputs_table_name, parameters=parameters), filtered_vids).fetchdf()
    # print("here1")
    # conn.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (vid, fid);")
    # print("select input videos: ", time.time() - _start)
    encountered_variables_prev_graphs = []
    encountered_variables_current_graph = []
    for graph_idx, dict in enumerate(current_query):
        # Generate scene graph:
        scene_graph = dict["scene_graph"]
        duration_constraint = dict["duration_constraint"]
        for p in scene_graph:
            for v in p["variables"]:
                if v not in encountered_variables_current_graph:
                    encountered_variables_current_graph.append(v)

        # Execute for unseen videos
        encountered_variables_current_graph = sorted(encountered_variables_current_graph, key=lambda x: int(x[1:]))
        table_list = ["obj_attr_filtered as {}".format(v) for v in encountered_variables_current_graph]
        where_clauses = []
        for i in range(len(encountered_variables_current_graph)-1):
            # [where condition] All obj_attr_filtered tables should have same vid and fid
            where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_current_graph[i], v2=encountered_variables_current_graph[i+1])) # join variables
        for p in scene_graph:
            predicate = p["predicate"]
            parameter = p.get("parameter", None) # only object predicates have parameters, which are class names
            variables = p["variables"]
            if predicate != "object":
                assert parameter is None
            if predicate == "object":
                # [where condition] object
                where_clauses.append(f"{variables[0]}.o1_oname = '{parameter}'")
            elif predicate in available_udf_names:
                if len(variables) == 1:
                    # [where condition] available attribute UDFs
                    where_clauses.append(f"'{predicate}' = ANY({variables[0]}.o1_anames)")
                else:
                    v0 = variables[0]
                    v1 = variables[1]
                    # [where condition] available relationship UDFs
                    pred_table = f"{v0}_{predicate}_{v1}"
                    table_list.append(f"rel_filtered as {pred_table}")
                    where_clauses.append(f"""
                        {v0}.vid = {pred_table}.vid
                        and {v0}.fid = {pred_table}.fid
                        and {v0}.o1_oid = {pred_table}.o1_oid
                        and {v1}.o1_oid = {pred_table}.o2_oid
                        and '{predicate}' = ANY({pred_table}.o1_o2_rnames)
                    """)
            elif f"udf_{predicate}" in materialized_udf_names:
                if len(variables) == 1:
                    # [where condition] new attribute UDFs (materialized)
                    v0 = variables[0]
                    pred_table = f"{v0}_{predicate}"
                    table_list.append(f"udf_{predicate} as {pred_table}")
                    where_clauses.append(f"""
                        {v0}.vid = {pred_table}.vid
                        and {v0}.fid = {pred_table}.fid
                        and {v0}.o1_oid = {pred_table}.o1_oid
                        and {pred_table}.pred = 1
                    """)
                else:
                    # [where condition] new relationship UDFs (materialized)
                    v0 = variables[0]
                    v1 = variables[1]
                    pred_table = f"{v0}_{predicate}_{v1}"
                    table_list.append(f"udf_{predicate} as {pred_table}")
                    where_clauses.append(f"""
                        {v0}.vid = {pred_table}.vid
                        and {v0}.fid = {pred_table}.fid
                        and {v0}.o1_oid = {pred_table}.o1_oid
                        and {v1}.o1_oid = {pred_table}.o2_oid
                        and {pred_table}.pred = 1
                    """)
            elif predicate in on_the_fly_udf_names:
                if len(variables) == 1:
                    # [where condition] new attribute UDFs (on-the-fly)
                    v0 = variables[0]
                    where_clauses.append(f"""
                        {predicate}({v0}.o1_oname, {v0}.o1_x1, {v0}.o1_y1, {v0}.o1_x2, {v0}.o1_y2, {v0}.o1_anames, {v0}.height, {v0}.width) = true
                    """)
                else:
                    # [where condition] new relationship UDFs (on-the-fly)
                    v0 = variables[0]
                    v1 = variables[1]
                    pred_table = f"{v0}_{predicate}_{v1}"
                    table_list.append(f"rel_filtered as {pred_table}")
                    where_clauses.append(f"""
                        {v0}.vid = {pred_table}.vid
                        and {v0}.fid = {pred_table}.fid
                        and {v0}.o1_oid = {pred_table}.o1_oid
                        and {v1}.o1_oid = {pred_table}.o2_oid
                        and {predicate}({pred_table}.o1_oname, {pred_table}.o1_x1, {pred_table}.o1_y1, {pred_table}.o1_x2, {pred_table}.o1_y2, {pred_table}.o1_anames, {pred_table}.o2_oname, {pred_table}.o2_x1, {pred_table}.o2_y1, {pred_table}.o2_x2, {pred_table}.o2_y2, {pred_table}.o2_anames, {pred_table}.o1_o2_rnames, {pred_table}.o2_o1_rnames, {pred_table}.height, {pred_table}.width) = true
                    """)
            else:
                raise ValueError("Unknown predicate: {}".format(predicate))
                # TODO: for robustness, we should remove it from the query and continue execution
        # [where condition] Different obj_attr_filtered tables should have different oids
        for var_pair in itertools.combinations(encountered_variables_current_graph, 2):
            where_clauses.append("{}.o1_oid <> {}.o1_oid".format(var_pair[0], var_pair[1]))
        # print("execute for unseen videos: ", time.time() - _start_execute)
        # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

        if graph_idx > 0:
            fields = "{v}.vid as vid, {v}.fid as fid".format(v=encountered_variables_current_graph[0])
            # fields += ", ".join(["{v}.o1_oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
            obj_union = copy.deepcopy(encountered_variables_prev_graphs)
            obj_union_fields = []
            obj_intersection_fields = []
            for v in encountered_variables_prev_graphs:
                obj_union_fields.append(f"t0.{v}_oid as {v}_oid")
            for v in encountered_variables_current_graph:
                if v in encountered_variables_prev_graphs:
                    obj_intersection_fields.append(f"t0.{v}_oid = {v}.o1_oid")
                else:
                    for u in encountered_variables_prev_graphs:
                        obj_intersection_fields.append(f"t0.{u}_oid <> {v}.o1_oid")
                    obj_union.append(v)
                    obj_union_fields.append(f"{v}.o1_oid as {v}_oid")
            obj_union_fields = ", ".join(obj_union_fields)
            obj_intersection_fields = " and ".join(obj_intersection_fields)
            where_clauses.append(f"""
                t0.vid = {encountered_variables_current_graph[0]}.vid
                AND {obj_intersection_fields}
                AND t0.fid < {encountered_variables_current_graph[0]}.fid
            """)
            where_clauses = " and ".join(where_clauses)
            table_list.append(f"g{graph_idx-1}_contiguous t0")
            table_str = ", ".join(table_list)
            sql_string = f"""
                CREATE TEMPORARY TABLE g{graph_idx} AS
                SELECT DISTINCT {fields}, {obj_union_fields}
                FROM {table_str}
                WHERE {where_clauses};
            """
            logger.debug(sql_string)
            conn.execute(sql_string)
            temp_tables.append(f"g{graph_idx}")
            # sql_string = """
            # CREATE TEMPORARY TABLE g{graph_idx}_filtered AS (
            #     SELECT DISTINCT t0.vid, t1.fid, {obj_union_fields}
            #     FROM g{graph_idx_prev}_contiguous t0, g{graph_idx} t1
            #     WHERE t0.vid = t1.vid AND {obj_intersection_fields} AND t0.fid < t1.fid
            # );
            # """.format(graph_idx=graph_idx, graph_idx_prev=graph_idx-1, obj_union_fields=obj_union_fields, obj_intersection_fields=obj_intersection_fields)
            # logger.debug(sql_string)
            # conn.execute(sql_string)
            # temp_tables.append(f"g{graph_idx}_filtered")
        else:
            fields = "{v}.vid as vid, {v}.fid as fid, ".format(v=encountered_variables_current_graph[0])
            fields += ", ".join(["{v}.o1_oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
            where_clauses = " and ".join(where_clauses)
            table_str = ", ".join(table_list)
            sql_string = f"""
                CREATE TEMPORARY TABLE g{graph_idx} AS
                SELECT DISTINCT {fields}
                FROM {table_str}
                WHERE {where_clauses};
            """
            logger.debug(sql_string)
            conn.execute(sql_string)
            temp_tables.append(f"g{graph_idx}")
            obj_union = encountered_variables_current_graph

        # Generate scene graph sequence:
        # table_name = "g{}_filtered".format(graph_idx) if graph_idx > 0 else "g{}".format(graph_idx)
        table_name = f"g{graph_idx}"
        obj_union_fields = ", ".join(["{}_oid".format(v) for v in obj_union])
        sql_string = """
            CREATE TEMPORARY TABLE g{graph_idx}_windowed AS (
            SELECT DISTINCT vid, fid, {obj_union_fields},
            lead(fid, {duration_constraint} - 1, 0) OVER (PARTITION BY vid, {obj_union_fields} ORDER BY fid) as fid_offset
            FROM {table_name}
        );
        """.format(graph_idx=graph_idx, duration_constraint=duration_constraint, obj_union_fields=obj_union_fields, table_name=table_name)
        logger.debug(sql_string)
        conn.execute(sql_string)
        temp_tables.append(f"g{graph_idx}_windowed")
        # print("windowed: ", time.time() - _start_windowed)

        sql_string = """
            CREATE TEMPORARY TABLE g{graph_idx}_contiguous AS (
            SELECT DISTINCT vid, {obj_union_fields}, min(fid_offset) AS fid
            FROM g{graph_idx}_windowed
            WHERE fid_offset = fid + ({duration_constraint} - 1)
            GROUP BY vid, {obj_union_fields}
        );
        """.format(graph_idx=graph_idx, obj_union_fields=obj_union_fields, duration_constraint=duration_constraint)
        logger.debug(sql_string)
        conn.execute(sql_string)
        temp_tables.append(f"g{graph_idx}_contiguous")
        conn.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(graph_idx))
        res = conn.fetchall()
        output_vids = [row[0] for row in res]
        encountered_variables_prev_graphs = obj_union
        encountered_variables_current_graph = []
    # Drop tables
    for temp_table in temp_tables:
        conn.execute(f"DROP TABLE {temp_table}")
    return output_vids
