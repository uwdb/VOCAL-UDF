import random
import string
import os
from vocaludf.utils import (
    duckdb_execute_cache_sequence,
    duckdb_execute_clevrer_cache_sequence,
    parse_signature,
)
from vocaludf.pretrained_model_api import image_captioning, image_classification, visual_question_answering, object_detection, image_segmentation, optical_character_recognition, depth_estimation
import time
import duckdb
import logging
from sklearn.metrics import f1_score

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class QueryExecutor:
    def __init__(self, config, inputs_table_name, registered_functions):
        self.config = config
        self.inputs_table_name = inputs_table_name
        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )
        self.registered_functions = registered_functions
        for func in self.registered_functions:
            signature = func["signature"]
            udf_name, _ = parse_signature(signature)
            lines = func["python_function"].split("\n")
            for i, line in enumerate(lines):
                if line.startswith('def '):
                    python_func_name = line.split()[1].split("(")[0]
                    # create a unique suffix
                    suffix = "".join(random.choices(string.ascii_lowercase, k=8))
                    python_func_args = line.split("(")[1].split(")")[0].split(",")
                    python_arg_str = ", ".join(f"{arg}: dict" for arg in python_func_args)
                    python_header_type_annotated = f"def {python_func_name}_{suffix}({python_arg_str}) -> bool:"
                    lines[i] = python_header_type_annotated
                    break
            # Rejoin the modified lines into a single string
            python_function = '\n'.join(lines)
            exec(python_function)
            exec(
                f"self.conn.create_function('{udf_name}', {python_func_name}_{suffix})"
            )
            logger.debug(f"Registered function: {signature}")

    def run(self, program, y_true, debug=False):
        if self.inputs_table_name == "Obj_clevrer":
            exec_func = duckdb_execute_clevrer_cache_sequence
            if debug:
                input_vids = 1000
            else:
                input_vids = 10000
        elif self.inputs_table_name == "Obj_clevr":
            exec_func = duckdb_execute_cache_sequence
            if debug:
                input_vids = 1500
            else:
                input_vids = 15000
        else:
            raise ValueError(
                "Unknown inputs_table_name: {}".format(self.inputs_table_name)
            )
        logger.info("Running query: {}".format(program["query"]))
        _start = time.time()
        memo = [{} for _ in range(72159)]  # Not used
        result, new_memo = exec_func(
            self.conn,
            program["query"],
            memo,
            self.inputs_table_name,
            input_vids,
            table_as_input_to_udf=True,
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