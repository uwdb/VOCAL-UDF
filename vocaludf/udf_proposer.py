from typing_extensions import Annotated
import autogen
import json
from typing import Tuple, List
import os
from vocaludf.utils import (
    replace_slot,
    parse_signature,
    transform_function
)
import time
import duckdb
import logging
from openai import OpenAI
import re
from collections import defaultdict
import importlib
import numpy as np
from sklearn.metrics import f1_score
import copy
import cv2

logging.basicConfig()
logger = logging.getLogger("vocal_udf")
logger.setLevel(logging.DEBUG)


class UDFCandidate:
    def __init__(self, id, payload):
        self.kwargs = payload.get("kwargs", {})
        self.id = str(id) + "_" + "_".join([f"{k}-{v}" for k, v in self.kwargs.items()]) if self.kwargs else str(id)
        self.udf_name = payload["udf_name"]
        self.udf_signature = payload["udf_signature"]
        self.udf_description = payload["udf_description"]
        self.semantic_interpretation = payload["semantic_interpretation"]
        self.function_implementation = payload["function_implementation"]
        self.score = 1  # F1 score
        self.test_score = -1
        self.loss_t = 0  # loss_t = n_misclassified

    def __str__(self):
        return f"UDFCandidate(id: {self.id}, function_implementation: {self.function_implementation}, test_score: {self.test_score}, score: {self.score}, loss_t: {self.loss_t})"


class UDFProposer:
    # Propose new UDFs and generate semantic interpretations
    def __init__(
        self,
        config,
        prompt_config,
        config_list,
        registered_functions,
        dataset,
        labeling_budget,
        num_interpretations,
        num_parameter_search,
        query_id,
        run_id,
        save_generated_udf,
        allow_kwargs_in_udf=False,
    ):
        self.config = config
        self.prompt_config = prompt_config
        self.config_list = config_list
        self.registered_functions = registered_functions
        self.dataset = dataset
        self.labeling_budget = labeling_budget
        self.num_interpretations = num_interpretations
        self.num_parameter_search = num_parameter_search
        self.query_id = query_id
        self.run_id = run_id
        self.save_generated_udf = save_generated_udf
        self.allow_kwargs_in_udf = allow_kwargs_in_udf

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

        self.udf_save_dir_lambda = lambda udf_name: os.path.join(
            self.config["output_dir"],
            "udf_generation",
            self.dataset,
            (
                "budget-{}_ninterp-{}_nparams-{}_with_kwargs".format(
                    self.labeling_budget, self.num_interpretations, self.num_parameter_search
                )
                if self.allow_kwargs_in_udf
                else "budget-{}_ninterp-{}_without_kwargs".format(
                    self.labeling_budget, self.num_interpretations
                )
            ),
            f"qid-{self.query_id}",
            f"run-{self.run_id}",
            udf_name,
        )

    def get_schema(self):
        object_schema = []
        relationship_schema = []
        attribute_schema = defaultdict(list)
        # Object schema: (obj_class_name, x1, y1, x2, y2)
        # Relationship schema: relationship_name
        # Attribute schema: key, possible values
        for registered_function in self.registered_functions:
            signature = registered_function["signature"]
            udf_name, udf_vars = parse_signature(signature)
            if len(udf_vars) == 2:
                # Relationship UDF
                relationship_schema.append(udf_name.lower())
            elif "_" in udf_name:
                # Attribute UDF
                udf_key, udf_value = udf_name.split("_")
                attribute_schema[udf_key.lower()].append(udf_value.lower())
            else:
                # Object UDF
                object_schema.append(udf_name.lower())
        # Construct a string that describes the schema
        # Merge the object schema and attribute schema into one, with attribute keys as the column names and attribute values as the possible values
        obj_cols = [
            ("class_name", "str"),
            ("x1", "float"),
            ("y1", "float"),
            ("x2", "float"),
            ("y2", "float"),
        ] + list((key, "str") for key in attribute_schema.keys())
        obj_cols_str = ", ".join([f"{col[0]}: {col[1]}" for col in obj_cols])
        # TODO: Add relationship to the prompt
        self.schema_prompt = (
            f"Each object is represented by a dictionary of {{ {obj_cols_str} }}. "
        )
        possible_values = ", ".join(["'" + c + "'" for c in object_schema])
        self.schema_prompt += (
            f"'class_name' can be one of the following: [{possible_values}]. "
        )
        image_h = self.config[self.dataset]["height"]
        image_w = self.config[self.dataset]["width"]
        # self.schema_prompt += f"x1, y1, x2, y2 are the top-left and bottom-right coordinates of the bounding box. "
        self.schema_prompt += f"x1, y1, x2, y2 are the top-left and bottom-right coordinates of the bounding box. The origin (x, y) = (0, 0) is located at the top left corner of a {image_w}x{image_h} frame. The x axis is oriented from left to right; the y axis is oriented from top to bottom. "
        for key, values in attribute_schema.items():
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
                        for func in self.registered_functions
                    ]
                )
            },
        )
        udf_proposer = autogen.AssistantAgent(
            name="udf_proposer",
            system_message=system_message,
            llm_config={
                "config_list": self.config_list,
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
        return self._implement(udf_signature, udf_description, with_pixels=False)

    def _implement(self, udf_signature, udf_description, with_pixels):
        udf_name, udf_vars = parse_signature(udf_signature)
        self.udf_save_dir = self.udf_save_dir_lambda(udf_name)
        os.makedirs(self.udf_save_dir, exist_ok=True)
        # Step 3: generate semantic interpretations and implement the UDF. Results are saved to disk
        generate_udfs_dict = self.prompt_config["generate_udfs"]
        if self.allow_kwargs_in_udf:
            if with_pixels:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["schema"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_with_optional_kwargs"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["schema"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation_with_optional_kwargs"]}'
            else:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["schema"]} {generate_udfs_dict["inputs_with_optional_kwargs"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_with_optional_kwargs"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["schema"]} {generate_udfs_dict["inputs_with_optional_kwargs"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation_with_optional_kwargs"]}'
        else:
            if with_pixels:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["schema"]} {generate_udfs_dict["inputs_with_pixels"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["schema"]} {generate_udfs_dict["inputs_with_pixels"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation"]}'
            else:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["schema"]} {generate_udfs_dict["inputs"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["schema"]} {generate_udfs_dict["inputs"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation"]}'
        n_obj = len(udf_vars)
        if with_pixels:
            udf_vars.insert(0, "img")
            actual_udf_signature = f"{udf_name}({', '.join(udf_vars)})"
        logger.info(
            f"Implementing UDF: {actual_udf_signature}, with {self.num_interpretations} semantic interpretations"
        )
        generate_udfs_prompt = replace_slot(
            generate_udfs_base_prompt,
            {
                "num_interpretations": self.num_interpretations,
                "udf_signature": actual_udf_signature,
                "udf_description": udf_description,
                "schema_info": self.schema_prompt,
                "n_obj": "one object" if n_obj == 1 else "two objects",
            },
        )
        logger.debug("generate_udfs_prompt: {}".format(generate_udfs_prompt))
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                logger.debug(f"trial: {trial}")
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
                implemented_udfs = eval(
                    "\n\n".join(
                        re.findall(
                            r"```json\n(.*?)```",
                            response.choices[0].message.content,
                            re.DOTALL,
                        )
                    )
                )["answer"][:self.num_interpretations]
                for idx in range(len(implemented_udfs)):
                    implemented_udf = implemented_udfs[idx]
                    implemented_udf = self.verify_syntax_correctness(implemented_udf, udf_name, actual_udf_signature, udf_description, n_obj, verify_syntax_correctness_base_prompt, with_pixels)
                    implemented_udf["udf_name"] = udf_name
                    implemented_udf["udf_signature"] = udf_signature
                    implemented_udf["udf_description"] = udf_description
                    implemented_udfs[idx] = implemented_udf
                    # implemented_udfs[idx] = copy.deepcopy(implemented_udf)
                    logger.info(f"[{idx}] semantic_interpretation: {implemented_udf['semantic_interpretation']}")
                    logger.info(f"[{idx}] function_implementation: {implemented_udf['function_implementation']}")
                    logger.info(f"[{idx}] kwargs: {implemented_udf.get('kwargs', {})}")
                    if self.save_generated_udf:
                        with open(os.path.join(self.udf_save_dir, f"{udf_name}_{idx}.json"), "w") as f:
                            json.dump(implemented_udf, f)
                break
            except Exception as e:
                logger.debug("ERROR: failed to implement UDF: {}".format(e))
                logger.debug(response)

        # Read UDF candidates from json files
        udf_candidate_list = []  # List[UDFCandidate]
        for i in range(self.num_interpretations):
            udf_dict = implemented_udfs[i]
            if self.allow_kwargs_in_udf and udf_dict.get("kwargs", {}):
                # Instantiate the kwargs with default values
                udf_variant_dict = copy.deepcopy(udf_dict)
                udf_variant_dict["kwargs"] = {k: v["default"] for k, v in udf_variant_dict["kwargs"].items() if v["default"] is not None}
                new_udf_candidate = UDFCandidate(id=i, payload=udf_variant_dict)
                logger.debug(new_udf_candidate)
                udf_candidate_list.append(new_udf_candidate)
                # Instantiate the kwargs with values randomly sampled from the range
                if self.num_parameter_search and self.num_parameter_search > 0:
                    for _ in range(self.num_parameter_search):
                        # deepcopy udf_dict
                        udf_variant_dict = copy.deepcopy(udf_dict)
                        for k in list(udf_variant_dict["kwargs"].keys()):
                            # randomly sample a value from the range
                            udf_variant_dict["kwargs"][k] = np.random.uniform(udf_variant_dict["kwargs"][k]["min"], udf_variant_dict["kwargs"][k]["max"])
                        new_udf_candidate = UDFCandidate(id=i, payload=udf_variant_dict)
                        logger.debug(new_udf_candidate)
                        udf_candidate_list.append(new_udf_candidate)
            else: # No additional arguments
                new_udf_candidate = UDFCandidate(id=i, payload=udf_dict)
                logger.debug(new_udf_candidate)
                udf_candidate_list.append(new_udf_candidate)
        return udf_candidate_list

    def verify_syntax_correctness(self, implemented_udf, udf_name, udf_signature, udf_description, n_obj, verify_syntax_correctness_base_prompt, with_pixels, n_verify_samples=10):
        df_samples = self.construct_train_and_test_data(n_obj, n_verify_samples)
        verify_syntax_correctness_prompt = replace_slot(
            verify_syntax_correctness_base_prompt,
            {
                "udf_signature": udf_signature,
                "udf_description": udf_description,
                "semantic_interpretation": implemented_udf["semantic_interpretation"],
                "schema_info": self.schema_prompt,
                "n_obj": "one object" if n_obj == 1 else "two objects",
            },
        )
        implemented_udf_json = json.dumps(implemented_udf)
        messages = [
            {"role": "user", "content": verify_syntax_correctness_prompt},
            {"role": "assistant", "content": "```json\n{}\n```".format(implemented_udf_json)}]
        for retry in range(5):
            try:
                if retry != 0:
                    response = self.client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=messages,
                        temperature=self.config["udf_generator"]["temperature"],
                        top_p=self.config["udf_generator"]["top_p"],
                        seed=self.run_id * 42,
                    )
                    messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    implemented_udf = eval(
                        "\n\n".join(
                            re.findall(
                                r"```json\n(.*?)```",
                                response.choices[0].message.content,
                                re.DOTALL,
                            )
                        )
                    )
                py_func_name = "py_{}".format(udf_name)
                namespace = {}
                exec(implemented_udf["function_implementation"], namespace)
                udf_obj = namespace[py_func_name]
                kwargs = {}
                for k, v in implemented_udf.get("kwargs", {}).items():
                    if v["default"] is not None:
                        kwargs[k] = float(v["default"])
                result = self.exec_udf_with_data(df_samples, udf_obj, kwargs, n_obj, with_pixels)
                contains_non_boolean = result.map(lambda x: not isinstance(x, bool)).any()
                if contains_non_boolean:
                    messages.append({"role": "user", "content": f"The function returned non-boolean value, but it should return a boolean value. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
                    continue
                break
            except Exception as e:
                messages.append({"role": "user", "content": f"Failed to execute the function due to the error: {type(e).__name__}: {e}. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
        if retry != 0:
            logger.debug("verify_syntax_correctness:\n" + "\n".join([f"{message['role']}: {message['content']}" for message in messages]))
        return implemented_udf

    def exec_udf_with_data(self, df, udf_obj, kwargs, n_obj, with_pixels):
        if n_obj == 1:
            if with_pixels:
                result = df.apply(
                    lambda row: udf_obj(row["img"], row["o1"], **kwargs), axis=1
                )
            else:
                result = df.apply(
                    lambda row: udf_obj(row["o1"], **kwargs), axis=1
                )
        elif n_obj == 2:
            if with_pixels:
                result = df.apply(
                    lambda row: udf_obj(row["img"], row["o1"], row["o2"], **kwargs), axis=1
                )
            else:
                result = df.apply(
                    lambda row: udf_obj(row["o1"], row["o2"], **kwargs), axis=1
                )
        return result

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
        self, udf_candidate_list, udf_name, df_train, n_obj, labeled_index, with_pixels
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

        indices_to_remove = []
        for i, udf_candidate in enumerate(udf_candidate_list):
            try:
                # For each sampled row in df_train, construct o1 and o2
                kwargs = {}
                for k, v in udf_candidate.kwargs.items():
                    kwargs[k] = float(v)
                py_func_name = "py_{}".format(udf_name)
                namespace = {}
                exec(udf_candidate.function_implementation, namespace)
                udf_obj = namespace[py_func_name]
                # TODO: may need to timeout if running for too long
                result = self.exec_udf_with_data(df_train.loc[sampled_index], udf_obj, kwargs, n_obj, with_pixels)
                contains_non_boolean = result.map(lambda x: not isinstance(x, bool)).any()
                if contains_non_boolean:
                    logger.debug(
                        f"ERROR: UDFCandidate(id={udf_candidate.id}) returned non-boolean value"
                    )
                    indices_to_remove.append(i)
                    continue
            except Exception as e:
                logger.debug(f"ERROR: failed to execute UDFCandidate(id={udf_candidate.id}): {e}")
                indices_to_remove.append(i)
                continue
            prediction_matrix.append(result.values)
        # Remove UDFs that failed to execute
        for i in sorted(indices_to_remove, reverse=True):
            del udf_candidate_list[i]

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
        self,
        gt_udf,
        udf_candidate,
        udf_name,
        n_obj,
        df,
        df_newly_labeled=None,
        with_pixels=False
    ):
        """
        Compute the F1 score of the UDF candidate on the data (train or test), using the ground truth UDF as the label
        if df_newly_labeled is provided, also compute the number of misclassified samples of them (which is used to compute loss_t)
        """
        try:
            # For each sampled row in df, construct o1 and o2
            kwargs = {}
            for k, v in udf_candidate.kwargs.items():
                kwargs[k] = float(v)
            py_func_name = "py_{}".format(udf_name)
            namespace = {}
            exec(udf_candidate.function_implementation, namespace)
            udf_obj = namespace[py_func_name]
            y_pred = self.exec_udf_with_data(df, udf_obj, kwargs, n_obj, with_pixels)
            if df_newly_labeled is not None:
                y_pred_new = self.exec_udf_with_data(df_newly_labeled, udf_obj, kwargs, n_obj, with_pixels)
        except Exception as e:
            logger.debug("ERROR: failed to execute udf_candidate {}: {}".format(udf_candidate.id, e))
            y_pred = df.apply(lambda row: False, axis=1)
            if df_newly_labeled is not None:
                y_pred_new = df_newly_labeled.apply(lambda row: False, axis=1)

        if n_obj == 1:
            y_true = df.apply(lambda row: gt_udf(row["o1"]), axis=1)
        elif n_obj == 2:
            y_true = df.apply(lambda row: gt_udf(row["o1"], row["o2"]), axis=1)
        # logger.debug(f"y_true: {y_true}, y_pred: {y_pred}")
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

    def construct_train_and_test_data(self, n_obj, n_train, n_test=None):
        # Construct training data and test data
        self.conn.execute(f"SELECT setseed({self.run_id / 100})")
        if n_obj == 1:
            df_filtered = self.conn.execute(
                "SELECT * FROM Obj_clevrer ORDER BY random() LIMIT {}".format(
                    n_train + n_test if n_test else n_train
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
                    project_clause, n_train + n_test if n_test else n_train
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
        if n_test:
            df_train = df_filtered[:n_train]
            df_train = df_train.reset_index()
            df_test = df_filtered[n_train:]
            df_test = df_test.reset_index()
            return df_train, df_test
        else:
            df_filtered = df_filtered.reset_index()
            return df_filtered

    def select(self, gt_udf_name, udf_candidate_list):
        return self._select(gt_udf_name, udf_candidate_list, with_pixels=False)

    def _select(self, gt_udf_name, udf_candidate_list, with_pixels):
        udf_signature = udf_candidate_list[0].udf_signature
        # Step 4: Select the best UDF
        udf_name, udf_vars = parse_signature(udf_signature)
        n_obj = len(udf_vars)

        # Construct training data and test data
        df_train, df_test = self.construct_train_and_test_data(n_obj, self.n_train, self.n_test)

        # Dynamically import the ground truth UDF
        # Example: gt_udf = gt_behind.gt_0
        module_name, function_name = gt_udf_name.split(".")
        module_name = "udfs.{}".format(module_name)
        module = importlib.import_module(module_name)
        gt_udf = getattr(module, function_name)

        # Select new video segments to label
        labeled_index = []
        segment_selection_time = 0
        _start_segment_selection_time = time.time()
        # TODO: perhaps regenerate one more UDF based on current labels after every k iterations
        for iter in range(self.labeling_budget):
            logger.info("iter {}".format(iter))
            _start_segment_selection_time_per_iter = time.time()
            new_labeled_index = self.select_sample(
                udf_candidate_list, udf_name, df_train, n_obj, labeled_index, with_pixels
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
                    udf_candidate_list[i],
                    udf_name,
                    n_obj,
                    df_train.loc[labeled_index],
                    df_train.loc[new_labeled_index],
                    with_pixels=with_pixels
                )
                udf_candidate_list[i].score = score
                udf_candidate_list[i].loss_t += loss_t
            # sort udf_candidate_list by score
            udf_candidate_list = sorted(
                udf_candidate_list, key=lambda x: x.score, reverse=True
            )
            logger.debug("updated udf_candidate_list: {}".format("\n".join([str(e) for e in udf_candidate_list])))
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
            udf_candidate_list[i].test_score = self.compute_udf_score(
                gt_udf,
                udf_candidate_list[i],
                udf_name,
                n_obj,
                df_test,
                with_pixels=with_pixels
            )
            logger.info(str(udf_candidate_list[i]))

        # compute the F1 score of the best udf (median F1 scores if there are multiple udfs with the same best score on the training set) on the test dataset
        best_score = max(udf_candidate.score for udf_candidate in udf_candidate_list)
        best_candidates = [
            udf_candidate
            for udf_candidate in udf_candidate_list
            if udf_candidate.score == best_score
        ]

        f1_score_test_list = []
        for best_candidate in best_candidates:
            f1_score_test_list.append(best_candidate.test_score)
        median_f1_score_test = np.median(f1_score_test_list)
        logger.info("median test f1: {}".format(median_f1_score_test))
        # TODO: If there are multiple best udfs, select the one with faster execution time?
        selected_udf_candidate = best_candidates[0]
        # Transforms the function by removing **kwargs from the function signature and replacing the kwargs with the actual values
        selected_udf_candidate.function_implementation = transform_function(
            original_code=selected_udf_candidate.function_implementation,
            instantiation_dict=selected_udf_candidate.kwargs,
        )
        if self.save_generated_udf:
            with open(os.path.join(self.udf_save_dir, f"{udf_name}_selected.json"), "w") as f:
                json.dump(
                    {
                        "udf_name": selected_udf_candidate.udf_name,
                        "udf_signature": selected_udf_candidate.udf_signature,
                        "udf_description": selected_udf_candidate.udf_description,
                        "semantic_interpretation": selected_udf_candidate.semantic_interpretation,
                        "function_implementation": selected_udf_candidate.function_implementation,
                        "f1_score_train": selected_udf_candidate.score,
                        "f1_score_test": selected_udf_candidate.test_score,
                    },
                    f,
                )
        logger.info(f"[Selected]: {str(selected_udf_candidate)}")
        return selected_udf_candidate

    def compute_best_test_score(self, gt_udf_name, udf_candidate_list):
        return self._compute_best_test_score(gt_udf_name, udf_candidate_list, with_pixels=False)

    def _compute_best_test_score(self, gt_udf_name, udf_candidate_list, with_pixels):
        udf_signature = udf_candidate_list[0].udf_signature
        # Step 4: Select the best UDF
        udf_name, udf_vars = parse_signature(udf_signature)
        n_obj = len(udf_vars)

        # Construct training data and test data
        _, df_test = self.construct_train_and_test_data(n_obj, self.n_train, self.n_test)

        # Dynamically import the ground truth UDF
        # Example: gt_udf = gt_behind.gt_0
        module_name, function_name = gt_udf_name.split(".")
        module_name = "udfs.{}".format(module_name)
        module = importlib.import_module(module_name)
        gt_udf = getattr(module, function_name)

        # compute test F1 score
        logger.info("compute test F1 score")
        best_test_score = -1
        for i in range(len(udf_candidate_list)):
            test_score = self.compute_udf_score(
                gt_udf,
                udf_candidate_list[i],
                udf_name,
                n_obj,
                df_test,
                with_pixels=with_pixels
            )
            udf_candidate_list[i].test_score = test_score
            logger.info(str(udf_candidate_list[i]))
            if test_score > best_test_score:
                best_test_score = test_score
        logger.info(f"[best_test_score]: {best_test_score}")
        return best_test_score

class CodeUDFWithPixelsProposer(UDFProposer):
    def __init__(
        self,
        config,
        prompt_config,
        config_list,
        registered_functions,
        dataset,
        labeling_budget,
        num_interpretations,
        num_parameter_search,
        query_id,
        run_id,
        save_generated_udf,
        allow_kwargs_in_udf=False,
    ):
        super().__init__(
            config,
            prompt_config,
            config_list,
            registered_functions,
            dataset,
            labeling_budget,
            num_interpretations,
            num_parameter_search,
            query_id,
            run_id,
            save_generated_udf,
            allow_kwargs_in_udf,
        )
        self.n_train = 1000
        self.n_test = 1000

        self.udf_save_dir_lambda = lambda udf_name: os.path.join(
            self.config["output_dir"],
            "compare_code_and_model_udf",
            self.dataset,
            (
                "budget-{}_ninterp-{}_nparams-{}_with_kwargs".format(
                    self.labeling_budget, self.num_interpretations, self.num_parameter_search
                )
                if self.allow_kwargs_in_udf
                else "budget-{}_ninterp-{}_without_kwargs".format(
                    self.labeling_budget, self.num_interpretations
                )
            ),
            f"run-{self.run_id}",
            udf_name,
        )

    def implement(self, udf_signature, udf_description):
        return self._implement(udf_signature, udf_description, with_pixels=True)

    def construct_train_and_test_data(self, n_obj, n_train, n_test=None):
        # Construct training data and test data
        # dataframe consists of columns: img, o1 [, o2]
        if n_test:
            df_train, df_test = super().construct_train_and_test_data(n_obj, n_train, n_test)
            # construct the img column
            for df in [df_train, df_test]:
                df["img"] = df.apply(self.frame_processing, axis=1)
            return df_train, df_test
        else:
            df = super().construct_train_and_test_data(n_obj, n_train)
            df["img"] = df.apply(self.frame_processing, axis=1)
            return df

    def frame_processing(self, row):
        vid = row.o1['vid']
        cap = cv2.VideoCapture(
            os.path.join(
                self.config['data_dir'],
                self.dataset,
                f'video_{str(vid//1000*1000).zfill(5)}-{str((vid//1000+1)*1000).zfill(5)}',
                f"video_{str(vid).zfill(5)}.mp4"
            )
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, row.o1['fid'])
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            raise ValueError("Failed to read the frame")
        cap.release()
        return frame

    def select(self, gt_udf_name, udf_candidate_list):
        return self._select(gt_udf_name, udf_candidate_list, with_pixels=True)

    def compute_best_test_score(self, gt_udf_name, udf_candidate_list):
        return self._compute_best_test_score(gt_udf_name, udf_candidate_list, with_pixels=True)