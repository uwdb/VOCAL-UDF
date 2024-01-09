from src.utils import program_to_dsl, dsl_to_program, postgres_execute, postgres_execute_cache_sequence, print_scene_graph_helper
import duckdb
import itertools
import copy

def duckdb_execute_cache_sequence(conn, current_query, memo, inputs_table_name, input_vids):
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
            parameter = p["parameter"]
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
        for var_pair in itertools.combinations(encountered_variables, 2):
            where_clauses.append("{}.oid <> {}.oid".format(var_pair[0], var_pair[1]))
        where_clauses = " and ".join(where_clauses)
        sql_sring = """SELECT DISTINCT {}.fid as fid FROM {} WHERE {};""".format(encountered_variables[0], tables, where_clauses)
        # print(sql_sring)
        cur.execute(sql_sring)
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