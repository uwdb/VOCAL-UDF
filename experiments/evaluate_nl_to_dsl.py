import yaml
import random
import json
from vocaludf.utils import setup_logging
import logging
import numpy as np
from vocaludf.query_executor import QueryExecutor
import argparse
import os
import sys
import resource
import duckdb

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

project_root = os.getenv("PROJECT_ROOT")

class QueryExecutorWithGroundTruth(QueryExecutor):
    def init_table(self):
        metadata_join_clause = '' if self.dataset in ['clevr', 'clevrer'] else f'LEFT OUTER JOIN {self.dataset}_metadata m ON o1.vid = m.vid AND o1.fid = m.fid'
        height_width_clause = '320 AS height, 480 AS width' if self.dataset in ['clevr', 'clevrer'] else 'm.height AS height, m.width AS width'
        group_by_clause = 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2' if self.dataset in ['clevr', 'clevrer'] else 'o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2, m.height, m.width'

        attr_parameters = ','.join('?' for _ in self.attribute_domain)
        sql = f"""
            CREATE TEMPORARY TABLE one_object AS
            SELECT
                o1.vid AS vid, o1.fid AS fid, o1.oid AS o1_oid, o1.oname AS o1_oname,
                o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                COALESCE(ARRAY_AGG(DISTINCT a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
                {height_width_clause}
            FROM {self.dataset}_objects o1
            LEFT OUTER JOIN {self.dataset}_attributes a ON o1.vid = a.vid AND o1.fid = a.fid AND o1.oid = a.oid
            {metadata_join_clause}
            GROUP BY {group_by_clause}
        """
        logger.debug(f"Create one_object table:\n{sql}")
        self.conn.execute(sql, self.attribute_domain).df()

        rel_parameters = ','.join('?' for _ in self.relationship_domain)
        sql = f"""
            CREATE TEMPORARY TABLE two_objects AS
            WITH obj_with_attrs AS (
                SELECT
                    o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                    COALESCE(ARRAY_AGG(DISTINCT a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS attributes
                FROM {self.dataset}_objects o
                LEFT OUTER JOIN {self.dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid AND a.aname = ANY([{attr_parameters}])
                GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
            )
            , relationships_expanded AS (
                SELECT
                    vid, fid, oid1, oid2,
                    COALESCE(ARRAY_AGG(DISTINCT rname) FILTER (WHERE rname = ANY([{rel_parameters}])), ARRAY[]::varchar[]) AS rnames
                FROM {self.dataset}_relationships
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
            JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid
            LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
            LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
            {metadata_join_clause}
            WHERE o1.oid <> o2.oid
        """
        logger.debug(f"Create two_objects table:\n{sql}")
        self.conn.execute(sql, self.attribute_domain + self.relationship_domain).df()

if __name__ == '__main__':
    # python evaluate_nl_to_dsl.py --query_id 0 --run_id 0 --dataset "clevrer" --query_filename "3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5 --num_workers 4 --n_train_distill 100 --selection_strategy "both" --pred_batch_size 4096 --dali_batch_size 1 --llm_method "gpt4v"
    config = yaml.safe_load(
        open(os.path.join(project_root, "configs", "config.yaml"), "r")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--query_filename", type=str, help="query filename")
    parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in UDF")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument("--n_selection_samples", type=int, default=500, help="number of sampled videos per selection iteration")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--program_with_pixels", action="store_true", help="program with pixels")
    parser.add_argument("--program_with_pretrained_models", action="store_true", help="program with pretrained models")
    parser.add_argument("--num_workers", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--pred_batch_size", type=int, default=262144, help="batch size for prediction data loader")
    parser.add_argument("--dali_batch_size", type=int, default=16, help="batch size for DALI")
    parser.add_argument("--llm_method", type=str, choices=["gpt", "llava"], default="gpt", help="LLM method for distill model annotations")

    args = parser.parse_args()
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    query_filename = args.query_filename
    allow_kwargs_in_udf = args.allow_kwargs_in_udf
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    n_selection_samples = args.n_selection_samples
    llm_method = args.llm_method
    num_interpretations = args.num_interpretations
    program_with_pixels = args.program_with_pixels
    program_with_pretrained_models = args.program_with_pretrained_models
    if program_with_pretrained_models:
        assert program_with_pixels, "program_with_pretrained_models requires program_with_pixels"
    num_workers = args.num_workers
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    pred_batch_size = args.pred_batch_size
    dali_batch_size = args.dali_batch_size

    config_name = "ninterp={}-nparams={}-kwargs={}-pixels={}-pretrained_models={}-ntrain_distill={}-nselection_samples={}-selection={}-budget={}-llm_method={}".format(
        num_interpretations,
        num_parameter_search,
        allow_kwargs_in_udf,
        program_with_pixels,
        program_with_pretrained_models,
        n_train_distill,
        n_selection_samples,
        selection_strategy,
        labeling_budget,
        llm_method,
    )

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_filename}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    positive_videos = input_query["positive_videos"]

    y_true = [1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"] // 2, config[dataset]["dataset_size"])]


    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    base_dir = os.path.join(
        "nl_to_dsl",
        dataset,
        query_filename,
        config_name,
    )
    log_filename = "qid={}-run={}.log".format(query_id, run_id)
    setup_logging(config, base_dir, log_filename, logger)

    with open(
        os.path.join(
            config["output_dir"],
            "udf_generation",
            dataset,
            query_filename,
            "num_missing_udfs=0",
            config_name,
            "qid={}-run={}.json".format(query_id, run_id),
        ),
        "r",
    ) as f:
        generation_output = json.load(f)
    program_with_pixels = generation_output["program_with_pixels"]
    parsed_program = generation_output["parsed_program"]
    object_domain = generation_output["object_domain"]
    relationship_domain = generation_output["relationship_domain"]
    attribute_domain = generation_output["attribute_domain"]
    registered_functions = generation_output["registered_functions"]
    available_udf_names = generation_output["available_udf_names"]
    materialized_df_names = generation_output["materialized_df_names"]
    on_the_fly_udf_names = generation_output["on_the_fly_udf_names"]

    logger.info("parsed_program: {}".format(parsed_program))

    qe = QueryExecutorWithGroundTruth(config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_df_names, on_the_fly_udf_names, program_with_pixels, num_workers, pred_batch_size, dali_batch_size)
    qe.run(parsed_program, y_true, debug=False)

    logger.info("Peak memory usuage (in GB): {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0))