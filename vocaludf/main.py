from typing_extensions import Annotated
import autogen
import yaml
import random
import string
import json
from typing import Tuple, List
from vocaludf.parser import parse, parse_udf
import pyparsing as pp
import os
from vocaludf.utils import (
    duckdb_execute_cache_sequence,
    duckdb_execute_clevrer_cache_sequence,
    replace_slot,
)
import time
import duckdb
import logging
import argparse
from openai import OpenAI
import re
from collections import defaultdict
import importlib
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics import f1_score

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

class UDFCandidate:
    def __init__(self, id, payload):
        self.id = id
        self.udf_name = payload["udf_name"]
        self.udf_signature = payload["udf_signature"]
        self.udf_description = payload["udf_description"]
        self.semantic_interpretation = payload["semantic_interpretation"]
        self.function_implementation = payload["function_implementation"]
        self.score = 1 # F1 score
        self.loss_t = 0 # loss_t = n_misclassified

    def __str__(self):
        return f"UDFCandidate(id: {self.id}, function_implementation: {self.function_implementation}, score: {self.score}, loss_t: {self.loss_t})"

class QueryParser:
    def __init__(
        self,
        config,
        prompt_config,
        config_list,
        dataset,
        registered_functions,
        run_id,
        allow_new_udfs=True,
    ):
        self.registered_function_names = []
        for registered_function in registered_functions:
            self.registered_function_names.append(
                registered_function["signature"].split("(")[0].lower()
            )
        if dataset in ["clevrer"]:  # video dataset
            dsl_definition_prompt = prompt_config["dsl_definition"]
        elif dataset in ["clevr"]:  # image dataset
            dsl_definition_prompt = prompt_config["dsl_definition_image"]
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        if allow_new_udfs:
            system_message = replace_slot(
                " ".join(
                    [
                        dsl_definition_prompt,
                        prompt_config["udf_definition"],
                        prompt_config["registered_udfs"],
                        prompt_config["parse_query"],
                    ]
                ),
                {
                    "functions": "\n".join(
                        [
                            "{}: {}".format(func["signature"], func["description"])
                            for func in registered_functions
                        ]
                    )
                },
            )
            is_termination_msg = lambda x: x.get("content", "") and (
                "yes" in x.get("content", "").rstrip().lower()
                or "no" in x.get("content", "").rstrip().lower()
            )
        else:
            system_message = replace_slot(
                " ".join(
                    [
                        dsl_definition_prompt,
                        prompt_config["udf_definition"],
                        prompt_config["registered_udfs"],
                        prompt_config["force_parse_query"],
                    ]
                ),
                {
                    "functions": "\n".join(
                        [
                            "{}: {}".format(func["signature"], func["description"])
                            for func in registered_functions
                        ]
                    )
                },
            )
            is_termination_msg = (
                lambda x: x.get("content", "")
                and "terminate" in x.get("content", "").rstrip().lower()
            )
        logger.debug("system_message: {}".format(system_message))

        self.parser = autogen.AssistantAgent(
            name="parser",
            system_message=system_message,
            llm_config={
                "config_list": config_list,
                "timeout": 120,
                "temperature": config["query_parser"]["temperature"],
                "seed": run_id,
            },
        )

        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=is_termination_msg,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": "coding", "use_docker": False},
        )

        @self.user_proxy.register_for_execution()
        @self.parser.register_for_llm(
            description="Verify syntax correctness of input query."
        )
        def verify_syntax(
            query: Annotated[str, "Input query written in DSL to be verified."]
        ) -> str:
            def check_UDF_support(program):
                unsupported_udfs = []
                flag = True
                for sg in program["query"]:
                    for pred in sg["scene_graph"]:
                        if (
                            pred["predicate"].lower()
                            not in self.registered_function_names
                        ):
                            flag = False
                            unsupported_udfs.append(pred["predicate"])
                return flag, unsupported_udfs

            try:
                parsed_program = parse().parseString(query, parseAll=True).as_dict()
                # Post-check if parsed program uses unsupported UDFs
                flag, unsupported_udfs = check_UDF_support(parsed_program)
                if flag:
                    self.parsed_program = parsed_program
                    self.parsed_query = query
                    return self.parsed_program
                else:
                    return (
                        query
                        + "failed:\n"
                        + "Unsupported UDFs: {}".format(unsupported_udfs)
                    )
            except (pp.ParseException, pp.ParseSyntaxException) as err:
                return err.explain()
            except Exception as e:
                error_message = query + "failed:\n" + str(e)
                return error_message

    def parse(self, user_query):
        self.user_proxy.initiate_chat(
            self.parser,
            message=user_query,
        )
        chat_messages = self.user_proxy.chat_messages[self.parser]
        logger.debug("chat_messages {}".format(chat_messages))
        if chat_messages[-1]["role"] != "user":
            # The conversation didn't end with the user's message (YES/NO)
            # Assume YES
            flag = "yes"
            logger.debug(
                "The conversation didn't end with the user's message. Assume: flag {}".format(
                    flag
                )
            )
        else:
            flag = chat_messages[-1]["content"].strip().lower()
            logger.debug("flag {}".format(flag))
        return flag

    def get_parsed_program(self):
        return self.parsed_program

    def get_parsed_query(self):
        return self.parsed_query


class UDFProposer:
    # Propose new UDFs and generate semantic interpretations
    def __init__(
        self,
        config,
        prompt_config,
        registered_functions,
        dataset,
        labeling_budget,
        num_interpretations,
        query_id,
        run_id,
    ):
        self.config = config
        self.prompt_config = prompt_config
        self.registered_functions = registered_functions
        self.dataset = dataset
        self.labeling_budget = labeling_budget
        self.num_interpretations = num_interpretations
        self.query_id = query_id
        self.run_id = run_id

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.get_schema()

        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )
        # Create a train and test split
        # NOTE: probably put these values in the config file
        self.n_train = 10000
        self.n_test = 10000

    def get_schema(self):
        self.object_schema = []
        self.relationship_schema = []
        self.attribute_schema = defaultdict(list)
        # Object schema: (obj_class_name, x1, y1, x2, y2)
        # Relationship schema: relationship_name
        # Attribute schema: key, possible values
        for registered_function in registered_functions:
            signature = registered_function["signature"]
            udf_name, udf_vars = parse_signature(signature)
            if len(udf_vars) == 2:
                # Relationship UDF
                self.relationship_schema.append(udf_name.lower())
            elif "_" in udf_name:
                # Attribute UDF
                udf_key, udf_value = udf_name.split("_")
                self.attribute_schema[udf_key.lower()].append(udf_value.lower())
            else:
                # Object UDF
                self.object_schema.append(udf_name.lower())
        # Construct a string that describes the schema
        # Merge the object schema and attribute schema into one, with attribute keys as the column names and attribute values as the possible values
        obj_cols = [
            ("class_name", "str"),
            ("x1", "float"),
            ("y1", "float"),
            ("x2", "float"),
            ("y2", "float"),
        ] + list((key, "str") for key in self.attribute_schema.keys())
        obj_cols_str = ", ".join([f"{col[0]}: {col[1]}" for col in obj_cols])
        # TODO: Add relationship to the prompt
        self.schema_prompt = (
            f"Each object is represented by a dictionary of {{ {obj_cols_str} }}. "
        )
        possible_values = ", ".join(["'" + c + "'" for c in self.object_schema])
        self.schema_prompt += (
            f"'class_name' can be one of the following: [{possible_values}]. "
        )
        image_h = self.config[self.dataset]["height"]
        image_w = self.config[self.dataset]["width"]
        # self.schema_prompt += f"x1, y1, x2, y2 are the top-left and bottom-right coordinates of the bounding box. "
        self.schema_prompt += f"x1, y1, x2, y2 are the top-left and bottom-right coordinates of the bounding box. The origin (x, y) = (0, 0) is located at the top left corner of a {image_w}x{image_h} frame. The x axis is oriented from left to right; the y axis is oriented from top to bottom. "
        for key, values in self.attribute_schema.items():
            possible_values = ", ".join(["'" + v + "'" for v in values])
            self.schema_prompt += (
                f"'{key}' can be one of the following: [{possible_values}]. "
            )
        logger.debug("Schema prompt: {}".format(self.schema_prompt))

    def propose(self, user_query):
        # Step 1: propose new UDFs
        logger.info("Proposing new UDFs")
        if self.dataset in ["clevrer"]:  # video dataset
            dsl_definition_prompt = self.prompt_config["dsl_definition"]
        elif self.dataset in ["clevr"]:  # image dataset
            dsl_definition_prompt = self.prompt_config["dsl_definition_image"]
        system_message = replace_slot(
            " ".join(
                [
                    dsl_definition_prompt,
                    self.prompt_config["udf_definition"],
                    self.prompt_config["registered_udfs"],
                    self.prompt_config["propose_udfs"],
                ]
            ),
            {
                "functions": "\n".join(
                    [
                        "{}: {}".format(func["signature"], func["description"])
                        for func in registered_functions
                    ]
                )
            },
        )
        udf_proposer = autogen.AssistantAgent(
            name="udf_proposer",
            system_message=system_message,
            llm_config={
                "config_list": config_list,
                "timeout": 120,
                "temperature": self.config["udf_proposer"]["temperature"],
                "seed": self.run_id,
                "top_p": self.config["udf_proposer"]["top_p"],
                "max_tokens": 512,
            },
        )

        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "")
            and "terminate" in x.get("content", "").rstrip().lower(),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": "coding", "use_docker": False},
        )

        @user_proxy.register_for_execution()
        @udf_proposer.register_for_llm(
            description="Verify syntax correctness of proposed UDFs."
        )
        def verify_syntax(
            proposed_functions: Annotated[
                List[List[str]],
                "A list of proposed functions where proposed_functions[i] = [signature_i, description_i]. 'signature_i' represents the function signature 'function(args)', and 'description_i' contains the function description.",
            ]
        ) -> str:
            try:
                invalid_funcs = []
                for proposed_function in proposed_functions:
                    signature = proposed_function[0]
                    udf_name, udf_vars = parse_signature(signature)
                    if len(udf_vars) != 1 and len(udf_vars) != 2:
                        invalid_funcs.append(signature)
                if len(invalid_funcs) > 0:
                    return f"Invalid number of arguments for proposed functions: {invalid_funcs}."
                else:
                    self.proposed_functions = {
                        func[0]: func[1] for func in proposed_functions
                    }
                    return "Success"
            except Exception as e:
                return "Error: " + str(e)

        logger.debug(f"system_message: {system_message}")

        user_proxy.initiate_chat(
            udf_proposer,
            message=f"User query: {user_query}",
        )

        logger.info(
            "Proposed functions: {}".format(self.proposed_functions)
        )  # key: signature, value: description
        registered_function_names = set(
            [
                registered_function["signature"].split("(")[0].lower()
                for registered_function in self.registered_functions
            ]
        )
        for key in list(self.proposed_functions.keys()):
            if key.split("(")[0].lower() in registered_function_names:
                del self.proposed_functions[key]
        # Step 2: verify functions (i.e., whether they can be constructed out of existing ones)
        # TODO: Implement this
        return self.proposed_functions

    def implement(self, udf_signature, udf_description):
        # Step 3: generate semantic interpretations and implement the UDF. Results are saved to disk
        # TODO: Consider kwargs
        generate_udfs_base_prompt = self.prompt_config["generate_udfs"]
        udf_name, udf_vars = parse_signature(udf_signature)
        logger.info(
            f"Implementing UDF: {udf_signature}, with {self.num_interpretations} semantic interpretations"
        )
        generate_udfs_prompt = replace_slot(
            generate_udfs_base_prompt,
            {
                "num_interpretations": self.num_interpretations,
                "udf_signature": udf_signature,
                "udf_description": udf_description,
                "schema_info": self.schema_prompt,
                "n_obj": "one object" if len(udf_vars) == 1 else "two objects",
            },
        )
        logger.debug("generate_udfs_prompt: {}".format(generate_udfs_prompt))
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert programming assistant.",
                        },
                        {"role": "user", "content": generate_udfs_prompt},
                    ],
                    temperature=self.config["udf_generator"]["temperature"],
                    top_p=self.config["udf_generator"]["top_p"],
                    seed=self.run_id * 42 + trial,
                )
                # NOTE: Sometimes GPT generates more UDFs than requested, so we remove the extra ones
                implemented_udfs = json.loads(
                    "\n\n".join(
                        re.findall(
                            r"```json\n(.*?)```",
                            response.choices[0].message.content,
                            re.DOTALL,
                        )
                    )
                )["answer"][:self.num_interpretations]

                os.makedirs(
                    os.path.join(
                        self.config["output_dir"],
                        "udf_generation",
                        self.dataset,
                        "budget-{}_ninterp-{}".format(
                            self.labeling_budget, self.num_interpretations
                        ),
                        f"qid-{self.query_id}",
                        f"run-{self.run_id}",
                        udf_name,
                    ),
                    exist_ok=True,
                )

                for idx, implemented_udf in enumerate(implemented_udfs):
                    semantic_interpretation = implemented_udf["semantic_interpretation"]
                    function_implementation = implemented_udf["function_implementation"]
                    with open(
                        os.path.join(
                            self.config["output_dir"],
                            "udf_generation",
                            self.dataset,
                            "budget-{}_ninterp-{}".format(
                                self.labeling_budget, self.num_interpretations
                            ),
                            f"qid-{self.query_id}",
                            f"run-{self.run_id}",
                            udf_name,
                            "{}_{}.json".format(udf_name, idx),
                        ),
                        "w",
                    ) as f:
                        json.dump(
                            {
                                "udf_name": udf_name,
                                "udf_signature": udf_signature,
                                "udf_description": udf_description,
                                "semantic_interpretation": semantic_interpretation,
                                "function_implementation": function_implementation,
                            },
                            f,
                        )
                    logger.info(
                        f"[{idx}] semantic_interpretation: {semantic_interpretation}"
                    )
                    logger.info(
                        f"[{idx}] function_implementation: {function_implementation}"
                    )
            except Exception as e:
                logger.debug("ERROR: failed to implement UDF: {}".format(e))
                logger.debug(response)

    def _compute_u_t(self, posterior_t, predictions_c):
        # Initialize possible u_t's
        u_t_list = np.zeros(2)

        # Repeat for each class
        for c in [0, 1]:
            # Compute the loss of models if the label of the streamed data is "c"
            loss_c = np.array(predictions_c != c) * 1
            # Compute the respective u_t value (conditioned on class c)
            term1 = np.inner(posterior_t, loss_c)
            u_t_list[c] = term1 * (1 - term1)

        # Return the final u_t
        u_t = np.max(u_t_list)

        return u_t

    def select_sample(
        self, udf_candidate_list, udf_name, df_train, n_obj, labeled_index
    ):
        # sample a subset of videos during each iteration
        n_sampled_videos = 500

        prediction_matrix = []
        _start = time.time()

        # query_list = [query_graph.program for query_graph, _ in self.candidate_queries[:self.pool_size]]
        # logger.debug("query pool", [program_to_dsl(query, self.rewrite_variables) for query in query_list])
        unlabeled_index = np.setdiff1d(
            np.arange(len(df_train)), labeled_index, assume_unique=True
        )
        logger.debug("len(unlabeled_index): {}".format(len(unlabeled_index)))

        # If more than n_sampled_videos videos, sample n_sampled_videos videos
        if len(unlabeled_index) > n_sampled_videos:
            sampled_index = np.random.choice(
                unlabeled_index, n_sampled_videos, replace=False
            )
        else:
            sampled_index = unlabeled_index

        for udf_candidate in udf_candidate_list:
            try:
                # For each sampled row in df_train, construct o1 and o2
                # TODO: kwargs
                kwargs = {}
                # for k, v in udf_candidate_dict["kwargs_signature"].items():
                #     kwargs[k] = float(v["default"])
                python_func_name = udf_candidate.function_implementation.split()[1].split("(")[0]
                namesapce = {}
                exec(udf_candidate.function_implementation, namesapce)
                udf_function = namesapce[f"py_{udf_name}"]
                if n_obj == 1:
                    result = df_train.loc[sampled_index].apply(
                        lambda row: udf_function(row["o1"], **kwargs), axis=1
                    )
                elif n_obj == 2:
                    result = df_train.loc[sampled_index].apply(
                        lambda row: udf_function(row["o1"], row["o2"], **kwargs), axis=1
                    )
            except Exception as e:
                logger.debug(f"ERROR: failed to execute UDFCandidate(id={udf_candidate.id}): {e}")
                result = df_train.loc[sampled_index].apply(lambda row: False, axis=1)
            prediction_matrix.append(result.values)

        prediction_matrix = np.array(
            prediction_matrix
        ).transpose()  # (n_samples, n_udfs)
        logger.debug(
            "constructing prediction matrix took {} seconds".format(
                time.time() - _start
            )
        )
        logger.debug("prediction_matrix size {}".format(prediction_matrix.shape))

        eta_0 = np.sqrt(np.log(len(udf_candidate_list)) / 2)

        # Use F1-scores as weights
        posterior_t = [udf_candidate.score for udf_candidate in udf_candidate_list]
        # Use the original weights as in the paper
        # eta = eta_0 / np.sqrt(n_sampled_videos)
        # loss_t = [loss_t for _, _, _, loss_t in udf_candidates_with_scores]
        # posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))

        posterior_t /= np.sum(posterior_t)  # normalized weight

        logger.debug("query weights {}".format(posterior_t))
        entropy_list = np.zeros(len(sampled_index))
        for i in range(len(sampled_index)):
            entropy_list[i] = self._compute_u_t(posterior_t, prediction_matrix[i, :])
        ind = np.argsort(-entropy_list)
        logger.debug("entropy list {}".format(entropy_list[ind]))
        # df_object_pairs_train[sampled_index[ind]].apply(lambda row: logger.info("o1: {}, o2: {}".format(row["o1"], row["o2"])), axis=1)
        logger.debug("sampled index {}".format(sampled_index[ind]))
        # find argmax of entropy (top k)
        max_entropy_index = sampled_index[np.argmax(entropy_list)]
        return [max_entropy_index]

    def compute_udf_score(
        self, gt_udf, function_implementation, udf_name, n_obj, df, df_newly_labeled=None
    ):
        """
        Compute the F1 score of the UDF candidate on the data (train or test), using the ground truth UDF as the label
        if df_newly_labeled is provided, also compute the number of misclassified samples of them (which is used to compute loss_t)
        """
        try:
            # For each sampled row in df, construct o1 and o2
            # TODO: kwargs
            kwargs = {}
            # for k, v in udf_candidate["kwargs_signature"].items():
            #     kwargs[k] = float(v["default"])

            namesapce = {}
            exec(function_implementation, namesapce)
            udf_function = namesapce[f"py_{udf_name}"]
            if n_obj == 1:
                y_pred = df.apply(lambda row: udf_function(row["o1"], **kwargs), axis=1)
            elif n_obj == 2:
                y_pred = df.apply(
                    lambda row: udf_function(row["o1"], row["o2"], **kwargs), axis=1
                )
            if df_newly_labeled is not None:
                if n_obj == 1:
                    y_pred_new = df_newly_labeled.apply(
                        lambda row: udf_function(row["o1"], **kwargs), axis=1
                    )
                elif n_obj == 2:
                    y_pred_new = df_newly_labeled.apply(
                        lambda row: udf_function(row["o1"], row["o2"], **kwargs), axis=1
                    )
        except Exception as e:
            # logger.debug("ERROR: failed to execute udf_candidate {}: {}".format(i, e))
            y_pred = df.apply(lambda row: False, axis=1)
            if df_newly_labeled is not None:
                y_pred_new = df_newly_labeled.apply(lambda row: False, axis=1)

        if n_obj == 1:
            y_true = df.apply(lambda row: gt_udf(row["o1"]), axis=1)
        elif n_obj == 2:
            y_true = df.apply(lambda row: gt_udf(row["o1"], row["o2"]), axis=1)
        logger.debug(f"y_true: {y_true}, y_pred: {y_pred}")
        score = f1_score(y_true, y_pred, zero_division=1.0)

        logger.info(
            "positive: {}, negative: {}".format(sum(y_true), len(y_true) - sum(y_true))
        )

        if df_newly_labeled is not None:
            if n_obj == 1:
                y_true_new = df_newly_labeled.apply(
                    lambda row: gt_udf(row["o1"]), axis=1
                )
            elif n_obj == 2:
                y_true_new = df_newly_labeled.apply(
                    lambda row: gt_udf(row["o1"], row["o2"]), axis=1
                )
            # Count the number of misclassifications for the new samples
            num_misclassified = np.sum(np.array(y_true_new != y_pred_new) * 1)
            return score, num_misclassified
        else:
            return score

    def select(self, udf_signature, udf_description, gt_udf_name):
        # Step 4: Select the best UDF
        udf_name, udf_vars = parse_signature(udf_signature)
        n_obj = len(udf_vars)

        # Construct training data and test data
        self.conn.execute(f"SELECT setseed({self.run_id / 100})")
        if n_obj == 1:
            df_filtered = self.conn.execute(
                "SELECT * FROM Obj_clevrer ORDER BY random() LIMIT {}".format(
                    self.n_train + self.n_test
                )
            ).df()
            df_filtered["o1"] = df_filtered.apply(lambda row: row.to_dict(), axis=1)
        elif n_obj == 2:
            schema = self.conn.execute("DESCRIBE Obj_clevrer").df()
            names = schema["column_name"].values
            # o1.vid as o1_vid, o1.fid as o1_fid,
            project_clause = (
                ", ".join(["o1.{} as o1_{}".format(name, name) for name in names])
                + ", "
                + ", ".join(["o2.{} as o2_{}".format(name, name) for name in names])
            )
            df_filtered = self.conn.execute(
                """
                SELECT {}
                FROM Obj_clevrer o1, Obj_clevrer o2
                WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
                ORDER BY random()
                LIMIT {}
            """.format(
                    project_clause, self.n_train + self.n_test
                )
            ).df()
            df_filtered["o1"] = df_filtered.apply(
                lambda row: {
                    col.split("_", 1)[1]: row[col]
                    for col in df_filtered.columns
                    if col.startswith("o1_")
                },
                axis=1,
            )
            df_filtered["o2"] = df_filtered.apply(
                lambda row: {
                    col.split("_", 1)[1]: row[col]
                    for col in df_filtered.columns
                    if col.startswith("o2_")
                },
                axis=1,
            )
        else:
            raise ValueError("Number of objects not supported: {}".format(n_obj))
        df_train = df_filtered[:self.n_train]
        df_train = df_train.reset_index()
        df_test = df_filtered[self.n_train:]
        df_test = df_test.reset_index()

        # Dynamically import the ground truth UDF
        # Example: gt_udf = gt_behind.gt_0
        module_name, function_name = gt_udf_name.split(".")
        module_name = "udfs.{}".format(module_name)
        module = importlib.import_module(module_name)
        gt_udf = getattr(module, function_name)

        # Read UDF candidates from json files
        udf_candidate_list = [] # List[UDFCandidate]
        for i in range(self.num_interpretations):
            with open(
                os.path.join(
                    self.config["output_dir"],
                    "udf_generation",
                    self.dataset,
                    "budget-{}_ninterp-{}".format(
                        self.labeling_budget, self.num_interpretations
                    ),
                    f"qid-{self.query_id}",
                    f"run-{self.run_id}",
                    udf_name,
                    "{}_{}.json".format(udf_name, i),
                ),
                "r",
            ) as f:
                new_udf_candidate = UDFCandidate(id=i, payload=json.load(f))
                logger.debug(new_udf_candidate)
                udf_candidate_list.append(new_udf_candidate)
                # TODO: kwargs
                # if len(udf_candidate["kwargs_signature"]) == 0:
                #     udf_candidates_with_scores.append([str(i), udf_candidate, 1, 0])
                # elif num_parameter_search <= 0:
                #     logger.info("use default values of kwargs in udf candidate {}".format(i))
                #     try:
                #         udf_id = str(i) + "_" + "_".join(["{}{}".format(k, v["default"]) for k, v in udf_candidate["kwargs_signature"].items()])
                #     except Exception as e:
                #         logger.debug("ERROR: failed to construct udf_id", e)
                #         logger.debug("udf_candidate {}".format(udf_candidate))
                #         udf_id = str(i)
                #     udf_candidates_with_scores.append([udf_id, udf_candidate, 1, 0])
                # else:
                #     for _ in range(num_parameter_search):
                #         # deepcopy udf_candidate
                #         udf_candidate_variant = copy.deepcopy(udf_candidate)
                #         for k, v in udf_candidate_variant["kwargs_signature"].items():
                #             # randomly sample a value from the range
                #             udf_candidate_variant["kwargs_signature"][k]["default"] = np.random.uniform(v["min"], v["max"])
                #         # create a unique udf identifier for each udf candidate by concatenating the ksys and the default values of the kwargs
                #         try:
                #             udf_id = str(i) + "_" + "_".join(["{}{}".format(k, v["default"]) for k, v in udf_candidate_variant["kwargs_signature"].items()])
                #         except Exception as e:
                #             logger.debug("ERROR: failed to construct udf_id", e)
                #             logger.debug("udf_candidate {}".format(udf_candidate))
                #             udf_id = str(i)
                #         udf_candidates_with_scores.append([udf_id, udf_candidate_variant, 1, 0])

        # Select new video segments to label
        labeled_index = []
        segment_selection_time = 0
        _start_segment_selection_time = time.time()
        for iter in range(self.labeling_budget):
            logger.info("iter {}".format(iter))
            _start_segment_selection_time_per_iter = time.time()
            new_labeled_index = self.select_sample(
                udf_candidate_list, udf_name, df_train, n_obj, labeled_index
            )
            logger.info("pick next segments {}".format(new_labeled_index))
            labeled_index += new_labeled_index
            logger.info("# labeled segments {}".format(len(labeled_index)))
            if n_obj == 1:
                y_true = df_train.loc[labeled_index].apply(
                    lambda row: gt_udf(row["o1"]), axis=1
                )
            elif n_obj == 2:
                y_true = df_train.loc[labeled_index].apply(
                    lambda row: gt_udf(row["o1"], row["o2"]), axis=1
                )
            # log number of positive and negative samples
            logger.info(
                "# positive: {}, # negative: {}".format(
                    sum(y_true), len(y_true) - sum(y_true)
                )
            )
            # Update scores
            for i in range(len(udf_candidate_list)):
                score, loss_t = self.compute_udf_score(
                    gt_udf,
                    udf_candidate_list[i].function_implementation,
                    udf_name,
                    n_obj,
                    df_train.loc[labeled_index],
                    df_train.loc[new_labeled_index],
                )
                udf_candidate_list[i].score = score
                udf_candidate_list[i].loss_t = loss_t
            # sort udf_candidate_list by score
            udf_candidate_list = sorted(
                udf_candidate_list, key=lambda x: x.score, reverse=True
            )
            logger.debug("updated udf_candidate_list: {}".format(udf_candidate_list))
            logger.debug(
                "test segment_selection_time_per_iter time: {}".format(
                    time.time() - _start_segment_selection_time_per_iter
                )
            )
        segment_selection_time += time.time() - _start_segment_selection_time
        logger.debug(
            "test segment_selection_time time: {}".format(segment_selection_time)
        )

        # compute test F1 score
        logger.info("compute test F1 score")
        for i in range(len(udf_candidate_list)):
            f1_score_test = self.compute_udf_score(
                gt_udf, udf_candidate_list[i].function_implementation, udf_name, n_obj, df_test
            )
            logger.info(
                "udf {}: test f1 {}, train f1 {}, n_misclassified {}".format(
                    udf_candidate_list[i].id,
                    f1_score_test,
                    udf_candidate_list[i].score,
                    udf_candidate_list[i].loss_t,
                )
            )

        # compute the F1 score of the best udf (median F1 scores if there are multiple udfs with the same best score on the training set) on the test dataset
        best_score = max(udf_candidate.score for udf_candidate in udf_candidate_list)
        best_candidates = [udf_candidate for udf_candidate in udf_candidate_list if udf_candidate.score == best_score]

        f1_score_test_list = []
        for best_candidate in best_candidates:
            f1_score_test = self.compute_udf_score(
                gt_udf, best_candidate.function_implementation, udf_name, n_obj, df_test
            )
            f1_score_test_list.append(f1_score_test)
        median_f1_score_test = np.median(f1_score_test_list)
        logger.info("median test f1: {}".format(median_f1_score_test))
        # TODO: If there are multiple best udfs, select the one with faster execution time?
        selected_udf_candidate = best_candidates[0]
        semantic_interpretation = selected_udf_candidate.semantic_interpretation
        function_implementation = selected_udf_candidate.function_implementation
        with open(
            os.path.join(
                self.config["output_dir"],
                "udf_generation",
                self.dataset,
                "budget-{}_ninterp-{}".format(
                    self.labeling_budget, self.num_interpretations
                ),
                f"qid-{self.query_id}",
                f"run-{self.run_id}",
                udf_name,
                "{}_selected.json".format(udf_name),
            ),
            "w",
        ) as f:
            json.dump(
                {
                    "udf_name": udf_name,
                    "udf_signature": udf_signature,
                    "udf_description": udf_description,
                    "semantic_interpretation": semantic_interpretation,
                    "function_implementation": function_implementation,
                    "f1_score_train": best_score,
                    "f1_score_test": f1_score_test_list[0],
                },
                f,
            )
        logger.info(f"[Selected] semantic_interpretation: {semantic_interpretation}")
        logger.info(f"[Selected] function_implementation: {function_implementation}")
        return selected_udf_candidate


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
            udf_name, udf_vars = parse_signature(signature)
            python_func_name = func["python_function"].split()[1].split("(")[0]
            # create a unique suffix
            suffix = "".join(random.choices(string.ascii_lowercase, k=8))
            python_func_args = (
                func["python_function"].split("(")[1].split(")")[0].split(",")
            )
            python_arg_str = ", ".join(f"{arg}: dict" for arg in python_func_args)
            python_header_type_annotated = (
                f"def {python_func_name}_{suffix}({python_arg_str}) -> bool:"
            )
            # Remove the first line of the function definition
            python_function = (
                python_header_type_annotated
                + "\n"
                + "\n".join(func["python_function"].split("\n")[1:])
            )
            exec(python_function)
            exec(
                f"self.conn.create_function('{udf_name}', {python_func_name}_{suffix})"
            )
            logger.debug(f"Registered function: {func['signature']}")

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


if __name__ == "__main__":
    # python main.py --query_id 0 --run_id 0 --dataset "clevrer" --budget 20 --num_interpretations 20
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )

    parser = argparse.ArgumentParser()
    # parser.add_argument("--udf_class", type=str, help="UDF class name we want to generate")
    # parser.add_argument("--n_obj", type=int, help="number of objects in the UDF signature")
    # parser.add_argument("--udf_description", type=str, help="UDF description")
    # parser.add_argument("--gt_udf_impl", type=str, help="ground truth UDF implementation")
    # parser.add_argument("--udf_generation_name", type=str, help="name of the function that GPT will generate")
    parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument(
        "--num_parameter_search",
        type=int,
        help="for udf candidate with kwargs, the number of different parameter values to explore",
    )
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument(
        "--ask_for_gt_udf",
        action="store_true",
        help="Ask for the gt_udf name interactively if enabled",
    )
    parser.add_argument(
        "--num_interpretations",
        type=int,
        help="number of semantic interpretations to generate for the UDF class",
    )
    # parser.add_argument("--stage", type=str, help="stage of the experiment (udf_generation, udf_selection, or all)")

    args = parser.parse_args()
    # gt_udf_impl = args.gt_udf_impl
    # udf_class = args.udf_class
    # n_obj= args.n_obj
    # udf_description = args.udf_description
    # udf_generation_name = "py_{}".format(udf_class)
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    ask_for_gt_udf = args.ask_for_gt_udf
    num_interpretations = args.num_interpretations
    # stage = args.stage

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = config[dataset]["input_query_file"]
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    gt_dsl = input_query["dsl"]
    user_query = input_query["question"]
    positive_videos = input_query["positive_videos"]
    y_true = [
        1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"])
    ]
    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    os.makedirs(
        os.path.join(config["log_dir"], "udf_generation", dataset), exist_ok=True
    )

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(
        os.path.join(
            config["log_dir"],
            "udf_generation",
            dataset,
            "qid-{}_run-{}-budget-{}_ninterp-{}.log".format(
                query_id, run_id, labeling_budget, num_interpretations
            ),
        ),
        mode="w",
    )
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Load LLM inference endpoints from an env variable or a file
    # See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
    # and OAI_CONFIG_LIST_sample
    config_list = autogen.config_list_from_json(
        env_or_file="/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        },
    )

    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )

    registered_functions = json.load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r")
    )["test"]

    # user_query = "Two objects move from far to close, then to far again"

    up = UDFProposer(
        config,
        prompt_config,
        registered_functions,
        dataset,
        labeling_budget,
        num_interpretations,
        query_id,
        run_id,
    )
    qp = QueryParser(
        config, prompt_config, config_list, dataset, registered_functions, run_id
    )
    flag = qp.parse(user_query)
    # Parse query
    # Step 1: propose new UDFs
    proposed_functions = up.propose(user_query)
    for udf_signature, udf_description in proposed_functions.items():
        # Step 2: generate semantic interpretations and implementations. Save the generated UDFs to disk
        up.implement(udf_signature, udf_description)
        # Step 3: Select the best UDF
        # TODO: perhaps regenerate one more UDF based on current labels after every k iterations
        # NOTE: If we use GPT-4 to provide feedback with zero user effort, how to incorporate the feedback into the UDF selection process?
        # First, retrieve the ground truth UDF
        if ask_for_gt_udf:
            # Ask the user for gt_udf name
            gt_udf_name = input(
                'Please enter gt_udf_name (options: "gt_near.gt_x", "gt_far.gt_x", "gt_rightof.gt_x", "gt_behind.gt_x", "gt_location_right.gt_x", "gt_location_bottom.gt_x", "gt_color_brown.gt_x", "gt_color_purple.gt_x", "gt_color_cyan.gt_x", "gt_color_yellow.gt_x", "gt_shape_cylinder.gt_x", "gt_material_metal.gt_x", where x is a non-negative integer): '
            )
        else:
            # HACK: Use a LM to automatically resolve the ground truth UDF
            # NOTE: Correctness is not guaranteed
            udf_name, udf_vars = parse_signature(udf_signature)
            if len(udf_vars) == 2:
                gt_udf_candidates = ["near", "far", "rightof", "behind"]
            else:
                gt_udf_candidates = [
                    "location_right",
                    "location_bottom",
                    "color_brown",
                    "color_purple",
                    "color_cyan",
                    "color_yellow",
                    "shape_cylinder",
                    "material_metal",
                ]
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            gt_udf_embeddings = model.encode(gt_udf_candidates)
            implemented_udf_embedding = model.encode([udf_name])
            similarities = util.pytorch_cos_sim(
                implemented_udf_embedding, gt_udf_embeddings
            )[0]
            gt_udf_name = "gt_{}.gt_0".format(gt_udf_candidates[similarities.argmax()])
            logger.debug(
                "similarities: {}".format(
                    [
                        f"{gt_udf_candidate}: {similarity}"
                        for gt_udf_candidate, similarity in zip(
                            gt_udf_candidates, similarities
                        )
                    ]
                )
            )
            logger.info(f"Selected gt_udf_name: {gt_udf_name}")
        selected_udf_candidate = up.select(udf_signature, udf_description, gt_udf_name)
        # Assume now that the best UDF is the first one
        # best_impl = implemented_udfs[0]
        logger.info(
            "Best {} implementation: {}".format(
                udf_signature, selected_udf_candidate.function_implementation
            )
        )
        # Step 5: Register the UDF
        new_udf = {
            "signature": udf_signature,
            "description": udf_description,
            "semantic_interpretation": selected_udf_candidate.semantic_interpretation,  # New field. Unsure if we need this
            "python_function": selected_udf_candidate.function_implementation,
        }
        registered_functions.append(new_udf)
    # Step 6: Re-parse the query
    # NOTE: Set allow_new_udfs=False. If the parser still wants to propose new UDFs, we will force it to generate a query that is the best approximation.
    qp = QueryParser(
        config,
        prompt_config,
        config_list,
        dataset,
        registered_functions,
        run_id,
        allow_new_udfs=False,
    )
    qp.parse(user_query)
    try:
        parsed_program = qp.get_parsed_program()
        parsed_dsl = qp.get_parsed_query()
        qe = QueryExecutor(config, f"Obj_{dataset}", registered_functions)
        qe.run(parsed_program, y_true, debug=False)
        # TODO: Only register UDFs that are actually used in the query and when the F1 score is above a certain threshold
    except Exception as e:
        logger.error("QueryExecutor Error: {}".format(e))
        logger.info("F1 score: 0")
