from collections import defaultdict
import time
import argparse
import logging
import yaml
import os
import json
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from openai import OpenAI, AsyncOpenAI
from vocaludf.utils import setup_logging, get_active_domain, parse_signature, duckdb_execute_video_materialize
import sys
import resource
import duckdb
from vocaludf.query_executor import QueryExecutor, remove_duplicates
from vocaludf.parser import parse

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

project_root = os.getenv("PROJECT_ROOT")

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s GB'''%(point, usage/1024.0/1024.0 )

class Gpt4oConceptQueryExecutor(QueryExecutor):
    def __init__(self, config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, sampling_rate, input_vids):
        self.test_data_init_time = 0
        self.query_execution_time = 0
        _start = time.time()
        self.config = config
        self.dataset = dataset
        self.object_domain = object_domain
        self.attribute_domain = attribute_domain
        self.relationship_domain = relationship_domain
        self.conn = duckdb.connect(
            database=os.path.join(config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )
        self.registered_functions = registered_functions
        self.available_udf_names = available_udf_names
        logger.info(f"available_udf_names: {self.available_udf_names}")
        self.sampling_rate = sampling_rate * 4 # The simplified setting already downsamples the frames by 4
        self.input_vids = input_vids
        self.init_table()
        self.test_data_init_time += time.time() - _start
        self.cost_estimation = 0

    def get_cost_estimation(self):
        return self.cost_estimation

    def init_table(self):
        metadata_join_clause = '' if self.dataset in ['clevrer'] else f'LEFT OUTER JOIN {self.dataset}_metadata m ON o1.vid = m.vid AND o1.fid = m.fid'
        height_width_clause = '320 AS height, 480 AS width' if self.dataset in ['clevrer'] else 'm.height AS height, m.width AS width'
        group_by_clause = 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2' if self.dataset in ['clevrer'] else 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2, m.height, m.width'

        attr_parameters = ','.join('?' for _ in self.attribute_domain)
        vid_parameters = ','.join('?' for _ in self.input_vids)
        if self.dataset == 'clevrer':
            where_clause = f"WHERE o1.vid = ANY([{vid_parameters}]) AND o1.fid % {self.sampling_rate} = 0"
        else:
            raise ValueError("Unknown dataset: {}".format(self.dataset))
        sql = f"""
            CREATE TEMPORARY TABLE one_object AS
            SELECT
                o1.vid AS vid, o1.fid // {self.sampling_rate} AS fid, o1.oid AS o1_oid, o1.oname AS o1_oname,
                o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                COALESCE(ARRAY_AGG(DISTINCT a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
                {height_width_clause}
            FROM {self.dataset}_objects o1
            LEFT OUTER JOIN {self.dataset}_attribute_predictions a ON o1.vid = a.vid AND o1.fid = a.fid AND o1.oid = a.oid
            {metadata_join_clause}
            {where_clause}
            GROUP BY {group_by_clause}
        """
        logger.debug(f"Create one_object table:\n{sql}")
        self.conn.execute(sql, self.attribute_domain + self.input_vids)

        rel_parameters = ','.join('?' for _ in self.relationship_domain)
        if self.dataset == 'clevrer':
            attr_where_clause = f"WHERE aname = ANY([{attr_parameters}]) AND vid = ANY([{vid_parameters}]) AND fid % {self.sampling_rate} = 0"
            obj_where_clause = f"WHERE vid = ANY([{vid_parameters}]) AND fid % {self.sampling_rate} = 0"
            rel_where_clause = f"WHERE rname = ANY([{rel_parameters}]) AND vid = ANY([{vid_parameters}]) AND fid % {self.sampling_rate} = 0"
        else:
            raise ValueError("Unknown dataset: {}".format(self.dataset))
        sql = f"""
            CREATE TEMPORARY TABLE two_objects AS
            WITH
                filtered_objects AS (
                    SELECT vid, fid // {self.sampling_rate} AS fid, oid, oname, x1, y1, x2, y2
                    FROM {self.dataset}_objects
                    {obj_where_clause}
                ),
                filtered_attributes AS (
                    SELECT vid, fid // {self.sampling_rate} AS fid, oid, aname
                    FROM {self.dataset}_attribute_predictions
                    {attr_where_clause}
                ),
                obj_with_attrs AS (
                    SELECT
                        o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                        COALESCE(ARRAY_AGG(DISTINCT a.aname), ARRAY[]::varchar[]) AS attributes
                    FROM filtered_objects o
                    LEFT OUTER JOIN filtered_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
                    GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
                ),
                relationships_expanded AS (
                    SELECT
                        vid, fid // {self.sampling_rate} AS fid, oid1, oid2,
                        COALESCE(ARRAY_AGG(DISTINCT rname), ARRAY[]::varchar[]) AS rnames
                    FROM {self.dataset}_relationship_predictions
                    {rel_where_clause}
                    GROUP BY vid, fid, oid1, oid2
                )
            SELECT
                o1.vid AS vid, o1.fid AS fid,
                o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2, o1.attributes AS o1_anames,
                o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2, o2.attributes AS o2_anames,
                COALESCE(r1.rnames, ARRAY[]::varchar[]) AS o1_o2_rnames,
                COALESCE(r2.rnames, ARRAY[]::varchar[]) AS o2_o1_rnames,
                {height_width_clause}
            FROM obj_with_attrs o1
            JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid <> o2.oid
            LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
            LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
            {metadata_join_clause}
        """
        logger.debug(f"Create two_objects table:\n{sql}")
        self.conn.execute(sql, self.input_vids + self.attribute_domain + self.input_vids + self.relationship_domain + self.input_vids)

    def run(self, program, y_true, debug=False):
        _start_run = time.time()
        # Bind each dataframe to a variable name, so that we can refer to them in the SQL query
        program["query"] = remove_duplicates(program["query"])
        logger.info("Running query: {}".format(program["query"]))

        _start = time.time()
        result = duckdb_execute_video_materialize(
            self.conn,
            program["query"],
            self.input_vids,
            available_udf_names=self.available_udf_names,
            materialized_udf_names=[],
            on_the_fly_udf_names=[],
        )
        logger.info("Time to execute query: {}".format(time.time() - _start))
        result = sorted(result)
        logger.info("output vids: {}".format(result))
        # logger.info("output vids: {}".format(result))
        y_pred = [1 if vid in result else 0 for vid in self.input_vids]

        f1 = f1_score(y_true[:len(y_pred)], y_pred)
        precision = precision_score(y_true[:len(y_pred)], y_pred)
        recall = recall_score(y_true[:len(y_pred)], y_pred)
        self.query_execution_time += time.time() - _start_run
        logger.info("Test data initialization time: {}".format(self.test_data_init_time))
        logger.info("Query execution time: {}".format(self.query_execution_time))
        logger.info("F1 score: {}".format(f1))
        logger.info("Precision: {}".format(precision))
        logger.info("Recall: {}".format(recall))
        return y_pred

def main():
    # python oracle_downsample_clevrer_simplified.py --query_id 0 --query_filename "simplified_3_new_udfs_labels" --sampling_rate 4
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_id', type=int, help='query id')
    parser.add_argument('--query_filename', type=str, help='query filename')
    parser.add_argument("--sampling_rate", type=int, help="sampling rate")
    args = parser.parse_args()
    query_id = args.query_id
    query_filename = args.query_filename
    sampling_rate = args.sampling_rate # 1, 2, 4, 8
    dataset = "clevrer"

    input_vids = [i for i in range(9500, 10000)]

    config = yaml.safe_load(open(os.path.join(project_root, "configs", "config.yaml"), "r"))

    # Set up logging
    base_dir = os.path.join(
        "oracle_clevrer_simplified",
        query_filename,
    )
    log_filename = "qid={}-sampling_rate={}.log".format(query_id, sampling_rate)
    setup_logging(config, base_dir, log_filename, logger)

    with open(os.path.join(config['data_dir'], "clevrer", f"{query_filename}.json"), "r") as f:
        input_query = json.load(f)['questions'][query_id]
    dsl = input_query["dsl"]
    y_true = [1 if i in input_query["positive_videos"] else 0 for i in input_vids]

    logger.info("vids: {}".format(input_vids))
    logger.info("y_true: {}".format(y_true))

    registered_udfs_json = json.load(open(os.path.join(project_root, "vocaludf", "registered_udfs.json"), "r"))
    registered_functions = registered_udfs_json[f"{dataset}_base"]
    new_modules = input_query["new_modules"]
    for new_module in new_modules:
        registered_functions.append(registered_udfs_json[dataset][new_module])
    logger.info("Registered functions: {}".format(registered_functions))
    object_domain, relationship_domain, attribute_domain = get_active_domain(config, dataset, registered_functions)
    logger.info("Active domains: object={}, relationship={}, attribute={}".format(object_domain, relationship_domain, attribute_domain))
    available_udf_names = [parse_signature(func["signature"])[0] for func in registered_functions]

    cost_estimation = defaultdict(float)
    total_execution_time = defaultdict(float)

    parsed_program = parse().parseString(dsl, parseAll=True).as_dict()

    # Step 4: Execute the query
    _start = time.time()
    qe = Gpt4oConceptQueryExecutor(
        config,
        dataset,
        object_domain,
        relationship_domain,
        attribute_domain,
        registered_functions,
        available_udf_names,
        sampling_rate,
        input_vids,
    )
    qe.run(parsed_program, y_true, debug=False)
    total_execution_time['query_execution'] += time.time() - _start
    cost_estimation['query_execution'] += qe.get_cost_estimation()

    logger.info("Total execution time: {}".format(total_execution_time))
    logger.info("Cost estimation: {}".format(cost_estimation))
    logger.info("Peak memory usuage (in GB): {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0))

if __name__ == "__main__":
    main()