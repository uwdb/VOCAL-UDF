import random
import string
import os
from vocaludf.utils import (
    duckdb_execute_cache_sequence,
    duckdb_execute_clevrer_cache_sequence,
    duckdb_execute_clevrer_materialize,
    parse_signature,
    PredImageDataset,
)
from vocaludf.pretrained_model_api import image_captioning, image_classification, visual_question_answering, object_detection, depth_estimation
import time
import duckdb
import logging
from sklearn.metrics import f1_score
from typing import List
import torch
from vocaludf import mlp
from torch.utils.data import IterableDataset
import math
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QueryExecutor:
    def __init__(self, config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_udf_names, on_the_fly_udf_names, num_workers):
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
        self.materialized_udf_names = materialized_udf_names
        self.on_the_fly_udf_names = on_the_fly_udf_names
        logger.info(f"available_udf_names: {self.available_udf_names}")
        logger.info(f"materialized_udf_names: {self.materialized_udf_names}")
        logger.info(f"on_the_fly_udf_names: {self.on_the_fly_udf_names}")
        self.attribute_features_dir = os.path.join(self.config[self.dataset]["features_dir"], "attribute")
        self.relationship_features_dir = os.path.join(self.config[self.dataset]["features_dir"], "relationship")
        self.num_workers = num_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.materialized_udfs = {}
        for func in self.registered_functions:
            signature = func["signature"]
            udf_name, _ = parse_signature(signature)
            if func.get("semantic_interpretation", "") == "model":
                df = self.get_materialized_df(func)
                self.materialized_udfs[udf_name] = df
            else:
                lines = func["function_implementation"].split("\n")
                for i, line in enumerate(lines):
                    if line.startswith('def '):
                        python_func_name = line.split()[1].split("(")[0]
                        # create a unique suffix
                        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
                        python_func_args = line.split("(")[1].split(")")[0].split(",")
                        # Add type annotations to the function header
                        # for attribute: attr(o1_oname: str, o1_x1: int, o1_y1: int, o1_x2: int, o1_y2: int, o1_anames: List[str], height: int, width: int)
                        # for relationship: rel(o1_oname: str, o1_x1: int, o1_y1: int, o1_x2: int, o1_y2: int, o1_anames: List[str], o2_oname: str, o2_x1: int, o2_y1: int, o2_x2: int, o2_y2: int, o2_anames: List[str], o1_o2_rnames: List[str], o2_o1_rnames: List[str], height: int, width: int)
                        if len(python_func_args) == 8:
                            # attribute
                            types = ["str", "int", "int", "int", "int", "List[str]", "int", "int"]
                        elif len(python_func_args) == 9:
                            # attribute + img
                            types = ["np.ndarray", "str", "int", "int", "int", "int", "List[str]", "int", "int"]
                        elif len(python_func_args) == 16:
                            # relationship
                            types = ["str", "int", "int", "int", "int", "List[str]", "str", "int", "int", "int", "int", "List[str]", "List[str]", "List[str]", "int", "int"]
                        elif len(python_func_args) == 17:
                            # relationship + img
                            types = ["np.ndarray", "str", "int", "int", "int", "int", "List[str]", "str", "int", "int", "int", "int", "List[str]", "List[str]", "List[str]", "int", "int"]
                        else:
                            raise ValueError("Unknown number of arguments in the function header: {}".format(len(python_func_args)))
                        python_arg_str = ", ".join([f"{arg}: {type}" for arg, type in zip(python_func_args, types)])
                        python_header_type_annotated = f"def {python_func_name}_{suffix}({python_arg_str}) -> bool:"
                        lines[i] = python_header_type_annotated
                        break
                # Rejoin the modified lines into a single string
                function_implementation = '\n'.join(lines)
                exec(function_implementation)
                exec(
                    f"self.conn.create_function('{udf_name}', {python_func_name}_{suffix})"
                )
            logger.debug(f"Registered function: {signature}")
        self.init_table()

    def init_table(self):
        attr_parameters = ','.join('?' for _ in self.attribute_domain)
        sql = f"""
            CREATE TEMPORARY TABLE one_object AS
            SELECT
                o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
                o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
                320 AS height, 480 AS width
            FROM {self.dataset}_objects o
            LEFT OUTER JOIN {self.dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
            GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
        """
        logger.debug(f"Create one_object table:\n{sql}")
        self.conn.execute(sql, self.attribute_domain).df()

        rel_parameters = ','.join('?' for _ in self.relationship_domain)
        sql = f"""
            CREATE TEMPORARY TABLE two_objects AS
            WITH obj_with_attrs AS (
                SELECT
                    o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                    COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS attributes
                FROM {self.dataset}_objects o
                LEFT OUTER JOIN {self.dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid AND a.aname = ANY([{attr_parameters}])
                GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
            )
            , relationships_expanded AS (
                SELECT
                    vid, fid, oid1, oid2,
                    COALESCE(ARRAY_AGG(rname) FILTER (WHERE rname = ANY([{rel_parameters}])), ARRAY[]::varchar[]) AS rnames,
                    ARRAY_AGG(rname) AS gt_rnames
                FROM {self.dataset}_relationships
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
        logger.debug(f"Create two_objects table:\n{sql}")
        self.conn.execute(sql, self.attribute_domain + self.relationship_domain).df()

    def get_materialized_df(self, func):
        best_ckpt = func["function_implementation"]
        # Predict the labels of all the data points
        checkpoint = torch.load(best_ckpt)
        hyper_parameters = checkpoint["hyper_parameters"]
        best_mlp_model = mlp.MLPProd(**hyper_parameters)
        best_mlp_model.load_state_dict(checkpoint["state_dict"])
        best_mlp_model.eval()
        best_mlp_model.to(self.device)

        signature = func["signature"]
        udf_name, udf_vars = parse_signature(signature)
        n_obj = len(udf_vars)

        pred_dataset = PredImageDataset(self.conn, n_obj, self.attribute_features_dir, self.relationship_features_dir)
        pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=256, num_workers=self.num_workers, shuffle=False)

        rows = defaultdict(list)
        predictions = []
        with torch.no_grad():
            for row, feature in tqdm(pred_loader):
                feature = feature.to(self.device)
                pred = best_mlp_model(feature)
                for k, v in row.items():
                    rows[k].extend(v.tolist())
                predictions.extend(pred.cpu().tolist())
                # logger.debug(using("profile"))
                # print("row: {}, pred: {}".format(row, pred.cpu().tolist()))
        # convert to tensor
        df_with_pred = pd.DataFrame(rows)
        df_with_pred['pred'] = predictions

        return df_with_pred

    def run(self, program, y_true, debug=False):
        # Bind each dataframe to a variable name, so that we can refer to them in the SQL query
        for udf_name, df in self.materialized_udfs.items():
            exec(f"{udf_name} = df")
        if self.dataset == "clevrer":
            exec_func = duckdb_execute_clevrer_materialize
            if debug:
                input_vids = 1000
            else:
                input_vids = 10000
        elif self.dataset == "clevr":
            exec_func = duckdb_execute_cache_sequence
            if debug:
                input_vids = 1500
            else:
                input_vids = 15000
        else:
            raise ValueError(
                "Unknown dataset: {}".format(self.dataset)
            )
        logger.info("Running query: {}".format(program["query"]))
        _start = time.time()
        memo = [{} for _ in range(72159)]  # Not used

        result, new_memo = exec_func(
            self.conn,
            program["query"],
            memo,
            input_vids,
            available_udf_names=self.available_udf_names,
            materialized_udf_names=self.materialized_udf_names,
            on_the_fly_udf_names=self.on_the_fly_udf_names
        )
        logger.info("Time to execute query: {}".format(time.time() - _start))
        result = sorted(result)
        # logger.info("output vids: {}".format(result))
        y_pred = [1 if i in result else 0 for i in range(input_vids)]

        # logger.info("predictions: {}".format(y_pred))
        # logger.info("true labels: {}".format(y_true[:input_vids]))
        f1 = f1_score(y_true[:input_vids], y_pred)
        logger.info("F1 score: {}".format(f1))
        return y_pred
