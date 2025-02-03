import asyncio
import json
import os
from PIL import Image
import time
import duckdb
import logging
import re
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import copy
import cv2
from tqdm import tqdm
import sys
import torch
from torch.utils.data import Dataset
import base64
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    after_log,
)  # for exponential backoff
import string
import lightning.pytorch as pl
from vocaludf import mlp
from vocaludf.utils import (
    replace_slot,
    parse_signature,
    PredImageDataset,
    MODEL_COST,
    expand_box,
    SharedResources,
    UtilsMixin,
    UDFCandidate,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CustomImageDataset(Dataset):
    def __init__(self, data, train):
        self.X = [d["image_features"] for d in data]
        if train:
            self.y = [d["llm_label"] for d in data]
        else:
            self.y = [d["label"] for d in data]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class UDFGenerator(UtilsMixin):
    mlp_method = "three_clip"
    def __init__(self, shared_resources: SharedResources, udf_signature, udf_description, gt_udf_name):
        # Shared resources
        self.shared_resources = shared_resources
        self.config = shared_resources.config
        self.prompt_config = shared_resources.prompt_config
        self.registered_functions = shared_resources.registered_functions
        self.object_domain = shared_resources.object_domain
        self.relationship_domain = shared_resources.relationship_domain
        self.attribute_domain = shared_resources.attribute_domain
        self.dataset = shared_resources.dataset
        self.labeling_budget = shared_resources.labeling_budget
        self.n_selection_samples = shared_resources.n_selection_samples
        self.num_interpretations = shared_resources.num_interpretations
        self.num_parameter_search = shared_resources.num_parameter_search
        self.program_with_pixels = shared_resources.program_with_pixels
        self.program_with_pretrained_models = shared_resources.program_with_pretrained_models
        self.query_class_name = shared_resources.query_class_name
        self.query_id = shared_resources.query_id
        self.run_id = shared_resources.run_id
        self.num_workers = shared_resources.num_workers
        self.selection_strategy = shared_resources.selection_strategy
        self.selection_labels = shared_resources.selection_labels
        self.allow_kwargs_in_udf = shared_resources.allow_kwargs_in_udf
        self.llm_method = shared_resources.llm_method
        self.is_async = shared_resources.is_async
        self.openai_model_name = shared_resources.openai_model_name
        self.client = shared_resources.client
        self.executor = shared_resources.executor
        self.n_train_selection = shared_resources.n_train_selection
        self.n_test_selection = shared_resources.n_test_selection
        self.n_train_distill = shared_resources.n_train_distill
        self.n_test_distill = shared_resources.n_test_distill
        self.save_labeled_data = shared_resources.save_labeled_data
        self.load_labeled_data = shared_resources.load_labeled_data
        self.attribute_features_dir = shared_resources.attribute_features_dir
        self.relationship_features_dir = shared_resources.relationship_features_dir
        self.one_object_df = shared_resources.one_object_df
        self.two_objects_df = shared_resources.two_objects_df
        self.device = shared_resources.device
        self.clip_model = shared_resources.clip_model
        self.clip_processor = shared_resources.clip_processor
        self.tokenizer = shared_resources.tokenizer
        self.dim_in = shared_resources.dim_in
        self.llava_model = shared_resources.llava_model
        self.llava_processor = shared_resources.llava_processor
        self.vid_to_vname = shared_resources.vid_to_vname

        # Per-UDF state variables
        self.udf_signature = udf_signature
        self.udf_description = udf_description
        self.gt_udf_name = gt_udf_name
        self.udf_name, self.udf_vars = parse_signature(udf_signature)
        self.n_obj = len(self.udf_vars)
        self.llm_positive_df = None
        self.llm_negative_df = None
        self.cost_estimation = defaultdict(float)
        self.execution_time = defaultdict(float)
        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )

    def get_cost_estimation(self):
        return self.cost_estimation

    def get_execution_time(self):
        return self.execution_time

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), after=after_log(logger, logging.DEBUG))
    async def completion_with_backoff(self, **kwargs):
        return await self.client.chat.completions.create(**kwargs)

    async def implement(self):
        udf_candidate_list = await self._implement()
        return udf_candidate_list, self.llm_positive_df, self.llm_negative_df

    async def _implement(self):
        if self.selection_strategy == "program":
            if self.selection_labels == "none" and self.allow_kwargs_in_udf:
                self.allow_kwargs_in_udf = False
                logger.info(f"[{self.udf_signature}] Turning off allow_kwargs_in_udf since no labels are provided")
            return await self._generate_program()
        elif self.selection_strategy == "model":
            return await self._distill_model()
        elif self.selection_strategy == "llm":
            if self.selection_labels == "none" and self.allow_kwargs_in_udf:
                self.allow_kwargs_in_udf = False
                logger.info(f"[{self.udf_signature}] Turning off allow_kwargs_in_udf since no labels are provided")
            llm_decision = await self._llm_decides_udf_type()
            if llm_decision == "programUDF":
                return await self._generate_program()
            elif llm_decision == "modelUDF":
                return await self._distill_model()
            else:
                raise NotImplementedError(f"llm_decision: {llm_decision} is not supported yet.")
        elif self.selection_strategy == "both":
            program_udf_candidates = await self._generate_program()
            model_udf_candidates = await self._distill_model()
            return program_udf_candidates + model_udf_candidates

    async def _llm_decides_udf_type(self):
        decide_udf_type_dict = self.prompt_config["decide_udf_type"]

        if self.program_with_pretrained_models:
            decide_udf_type_base_prompt = f'{decide_udf_type_dict["overall"]} {decide_udf_type_dict["code_with_pixels_and_pretrained_models"]} {decide_udf_type_dict["model"]} {decide_udf_type_dict["output"]}'
        elif self.program_with_pixels:
            decide_udf_type_base_prompt = f'{decide_udf_type_dict["overall"]} {decide_udf_type_dict["code_with_pixels"]} {decide_udf_type_dict["model"]} {decide_udf_type_dict["output"]}'
        else:
            decide_udf_type_base_prompt = f'{decide_udf_type_dict["overall"]} {decide_udf_type_dict["code_without_pixels"]} {decide_udf_type_dict["model"]} {decide_udf_type_dict["output"]}'

        decide_udf_type_prompt = replace_slot(
            decide_udf_type_base_prompt,
            {
                "udf_description": self.udf_description,
                "available_concepts": self.object_domain + self.relationship_domain + self.attribute_domain,
            },
        )
        logger.debug(f"[{self.udf_signature}] decide_udf_type_prompt: {decide_udf_type_prompt}")
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                logger.debug(f"[{self.udf_signature}] trial: {trial}")
                response = await self.completion_with_backoff(
                    model=self.openai_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert programming assistant.",
                        },
                        {"role": "user", "content": decide_udf_type_prompt},
                    ],
                    temperature=self.config["udf_generator"]["temperature"],
                    top_p=self.config["udf_generator"]["top_p"],
                    seed=self.run_id * 42 + trial,
                )
                llm_decision = response.choices[0].message.content
                self.cost_estimation["decide_udf_type"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
                logger.debug(f"[{self.udf_signature}] llm_decision: {llm_decision}")
                if "programUDF" in llm_decision:
                    return "programUDF"
                elif "modelUDF" in llm_decision:
                    return "modelUDF"
            except Exception as e:
                logger.exception(f"[{self.udf_signature}] ERROR: failed to decide UDF type: {e}")
                logger.debug(f"[{self.udf_signature}] {response}")


    ##############################################################
    #############                                    #############
    ############# UDF implementation (program-based) #############
    #############                                    #############
    ##############################################################
    async def _generate_program(self):
        """
        Implements the UDF program based on the given UDF signature and description.

        Args:
            udf_signature (str): The signature of the UDF.
            udf_description (str): The description of the UDF.

        Returns:
            list: A list of UDFCandidate objects representing the implemented UDFs.
        """
        _start = time.time()
        logger.info(f"[{self.udf_signature}] Program generation started")
        # Generate semantic interpretations and implement the UDF.
        generate_udfs_dict = self.prompt_config["generate_udfs"]
        attr_or_rel = "attribute" if self.n_obj == 1 else "relationship"
        if self.allow_kwargs_in_udf:
            if self.program_with_pretrained_models: # Unused as they are expensive
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_with_optional_kwargs"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation_with_optional_kwargs"]}'
            elif self.program_with_pixels:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_with_optional_kwargs"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation_with_optional_kwargs"]}'
            else:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_with_optional_kwargs"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation_with_optional_kwargs"]}'
        else:
            if self.program_with_pretrained_models:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_pixels"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_pixels"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation"]}'
            elif self.program_with_pixels:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_pixels"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_pixels"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation"]}'
            else:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation"]}'
        # Construct python function arguments
        if self.n_obj == 1:
            py_func_args = [f"{self.udf_vars[0]}_oname", f"{self.udf_vars[0]}_x1", f"{self.udf_vars[0]}_y1", f"{self.udf_vars[0]}_x2", f"{self.udf_vars[0]}_y2", f"{self.udf_vars[0]}_anames", "height", "width"]
        else:
            py_func_args = [f"{self.udf_vars[0]}_oname", f"{self.udf_vars[0]}_x1", f"{self.udf_vars[0]}_y1", f"{self.udf_vars[0]}_x2", f"{self.udf_vars[0]}_y2", f"{self.udf_vars[0]}_anames", f"{self.udf_vars[1]}_oname", f"{self.udf_vars[1]}_x1", f"{self.udf_vars[1]}_y1", f"{self.udf_vars[1]}_x2", f"{self.udf_vars[1]}_y2", f"{self.udf_vars[1]}_anames", f"{self.udf_vars[0]}_{self.udf_vars[1]}_rnames", f"{self.udf_vars[1]}_{self.udf_vars[0]}_rnames", "height", "width"]
        if self.allow_kwargs_in_udf:
            py_func_args.append("**kwargs")
        if self.program_with_pixels:
            py_func_args.insert(0, "img")
        py_func_signature = f"{self.udf_name}({', '.join(py_func_args)})"
        logger.info(
            f"[{self.udf_signature}] Implementing UDF: {py_func_signature}, with {self.num_interpretations} semantic interpretations"
        )
        generate_udfs_prompt = replace_slot(
            generate_udfs_base_prompt,
            {
                "num_interpretations": self.num_interpretations,
                "udf_signature": py_func_signature,
                "udf_description": self.udf_description,
                "object_domain": self.object_domain,
                "relationship_domain": self.relationship_domain,
                "attribute_domain": self.attribute_domain,
                "o1": self.udf_vars[0],
                "o2": self.udf_vars[1] if self.n_obj == 2 else "",
            },
        )
        logger.debug(f"[{self.udf_signature}] generate_udfs_prompt: {generate_udfs_prompt}")
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                logger.debug(f"[{self.udf_signature}] trial: {trial}")
                self.execution_time["program_generation"] += time.time() - _start
                response = await self.completion_with_backoff(
                    model=self.openai_model_name,
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
                _start = time.time()
                self.cost_estimation["generate_program"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
                # NOTE: Sometimes LLM generates more UDFs than requested, so we remove the extra ones
                verifed_implemented_udfs = []
                implemented_udfs = eval(
                    "\n\n".join(
                        re.findall(
                            r"```json\n(.*?)```",
                            response.choices[0].message.content,
                            re.DOTALL,
                        )
                    )
                )["answer"][:self.num_interpretations]
                logger.debug(f"[{self.udf_signature}] implemented_udfs: {implemented_udfs}")
                for idx in range(len(implemented_udfs)):
                    implemented_udf = implemented_udfs[idx]
                    self.execution_time["program_generation"] += time.time() - _start
                    implemented_udf, success = await self.verify_syntax_correctness(implemented_udf, self.udf_vars, self.udf_name, py_func_signature, py_func_args, self.udf_description, self.n_obj, verify_syntax_correctness_base_prompt)
                    _start = time.time()
                    if success:
                        implemented_udf["udf_name"] = self.udf_name
                        implemented_udf["udf_signature"] = self.udf_signature
                        implemented_udf["udf_description"] = self.udf_description
                        verifed_implemented_udfs.append(implemented_udf)
                        logger.info(f"[{self.udf_signature}] [{idx}] semantic_interpretation: {implemented_udf['semantic_interpretation']}")
                        logger.info(f"[{self.udf_signature}] [{idx}] function_implementation: {implemented_udf['function_implementation']}")
                        logger.info(f"[{self.udf_signature}] [{idx}] kwargs: {implemented_udf.get('kwargs', {})}")
                break
            except Exception as e:
                logger.exception(f"[{self.udf_signature}] ERROR: failed to implement UDF: {e}")
                logger.debug(f"[{self.udf_signature}] {response}")

        # Read UDF candidates from json files
        udf_candidate_list = []  # List[UDFCandidate]
        for i in range(len(verifed_implemented_udfs)):
            try:
                udf_dict = verifed_implemented_udfs[i]
                if self.allow_kwargs_in_udf and udf_dict.get("kwargs", {}):
                    # Instantiate the kwargs with default values
                    udf_variant_dict = copy.deepcopy(udf_dict)
                    udf_variant_dict["kwargs"] = {k: v["default"] for k, v in udf_variant_dict["kwargs"].items() if v["default"] is not None}
                    new_udf_candidate = UDFCandidate(id=i, payload=udf_variant_dict)
                    logger.debug(f"[{self.udf_signature}] {new_udf_candidate}")
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
                            logger.debug(f"[{self.udf_signature}] {new_udf_candidate}")
                            udf_candidate_list.append(new_udf_candidate)
                else: # No additional arguments
                    new_udf_candidate = UDFCandidate(id=i, payload=udf_dict)
                    logger.debug(f"[{self.udf_signature}] {new_udf_candidate}")
                    udf_candidate_list.append(new_udf_candidate)
            except Exception as e:
                logger.exception(f"[{self.udf_signature}] Failed to read UDF candidate: {e}")
        logger.info(f"[{self.udf_signature}] Program generation finished")
        self.execution_time["program_generation"] += time.time() - _start
        return udf_candidate_list

    async def verify_syntax_correctness(self, implemented_udf, udf_vars, udf_name, py_func_signature, py_func_args, udf_description, n_obj, verify_syntax_correctness_base_prompt, n_verify_samples=10):
        _start = time.time()
        df_samples = self.construct_train_and_test_data(n_obj, n_verify_samples, df_with_img_column=self.program_with_pixels)
        verify_syntax_correctness_prompt = replace_slot(
            verify_syntax_correctness_base_prompt,
            {
                "udf_signature": py_func_signature,
                "udf_description": udf_description,
                "semantic_interpretation": implemented_udf["semantic_interpretation"],
                "object_domain": self.object_domain,
                "relationship_domain": self.relationship_domain,
                "attribute_domain": self.attribute_domain,
                "o1": udf_vars[0],
                "o2": udf_vars[1] if n_obj == 2 else "",
            },
        )
        implemented_udf_json = json.dumps(implemented_udf)
        messages = [
            {"role": "user", "content": verify_syntax_correctness_prompt},
            {"role": "assistant", "content": "```json\n{}\n```".format(implemented_udf_json)}]
        success = False
        for retry in range(5):
            try:
                if retry != 0:
                    self.execution_time["program_generation"] += time.time() - _start
                    response = await self.completion_with_backoff(
                        model=self.openai_model_name,
                        messages=messages,
                        temperature=self.config["udf_generator"]["temperature"],
                        top_p=self.config["udf_generator"]["top_p"],
                        seed=self.run_id * 42,
                    )
                    _start = time.time()
                    self.cost_estimation["verify_syntax_correctness"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
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
                # Verify if the function has the correct number of arguments
                is_header_correct = True
                import_sklearn = False
                lines = implemented_udf["function_implementation"].split("\n")
                for _, line in enumerate(lines):
                    if line.startswith('def '):
                        generated_py_func_args = [arg.strip() for arg in line.split("(")[1].split(")")[0].split(",")]
                        if len(generated_py_func_args) != len(py_func_args):
                            messages.append({"role": "user", "content": f"Expected {len(py_func_args)} arguments, but got {len(generated_py_func_args)}. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
                            is_header_correct = False
                            break
                        for i, (gt_arg, generated_arg) in enumerate(zip(py_func_args, generated_py_func_args)):
                            if gt_arg != generated_arg:
                                messages.append({"role": "user", "content": f"Expected {gt_arg} as argument #{i+1}, but got {generated_arg}. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
                                is_header_correct = False
                                break
                    elif 'import' in line and 'sklearn' in line:
                        messages.append({"role": "user", "content": f"Using sklearn in multithreading environments may cause deadlock. Please do not use sklearn library. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
                        import_sklearn = True
                if (not is_header_correct) or import_sklearn:
                    continue
                py_func_name = "py_{}".format(udf_name)
                exec(implemented_udf["function_implementation"], globals())
                udf_obj = globals()[py_func_name]
                is_kwargs_correct = True
                kwargs = {}
                for k, v in implemented_udf.get("kwargs", {}).items():
                    if v["default"] is not None:
                        try:
                            kwargs[k] = float(v["default"])
                            v["min"] = float(v["min"])
                            v["max"] = float(v["max"])
                        except Exception as e:
                            messages.append({"role": "user", "content": f"Failed to parse kwargs due to the error: {type(e).__name__}: {e}. Please fix it and regenerate 'kwargs' using the same 'semantic_interpretation' and 'function_implementation'."})
                            is_kwargs_correct = False
                            break
                if not is_kwargs_correct:
                    continue
                result = self.exec_udf_with_data(df_samples, udf_obj, kwargs, n_obj, timeout=60)
                contains_non_boolean = False
                for r in result:
                    if r != 1 and r != 0:
                        contains_non_boolean = True
                        break
                if contains_non_boolean:
                    logger.debug(f"[{self.udf_signature}] The function returned non-boolean value: {result}")
                    messages.append({"role": "user", "content": f"The function returned non-boolean value, but it should return a boolean value. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
                    continue
                success = True
                break
            except Exception as e:
                messages.append({"role": "user", "content": f"Failed to execute the function due to the error: {type(e).__name__}: {e}. Please fix it and regenerate 'function_implementation' using the same 'semantic_interpretation'."})
        if retry != 0:
            logger.debug(f"[{self.udf_signature}] verify_syntax_correctness:\n" + "\n".join([f"{message['role']}: {message['content']}" for message in messages]))
        self.execution_time["program_generation"] += time.time() - _start
        return implemented_udf, success


    ##############################################################
    #############                                    #############
    ############# UDF Distillation (distilled-model) #############
    #############                                    #############
    ##############################################################
    async def _distill_model(self):
        """
        gt_udf_name: ground truth UDF name. If provided, compute llm's TP, FP, TN, FN, f1 score and test the trained model.
        """
        logger.info(f"[{self.udf_signature}] Model distillation started")

        logger.info(f"[{self.udf_signature}] Model distillation (initialization) started")
        _start = time.time()
        attribute_df = self.conn.execute(f"SELECT * FROM {self.dataset}_attributes").df()
        relationship_df = self.conn.execute(f"SELECT * FROM {self.dataset}_relationships").df()

        # ask LLM about relevant object classes to the target relationships, and filter data
        filtered_objects, filtered_subjects, filtered_targets = None, None, None
        self.execution_time["model_distillation_init"] += time.time() - _start
        if self.dataset in ["charades"]:
            filtered_objects = list(set(await self.llm_filter_relevant_objects(self.udf_signature, self.udf_description) + ['person']))
        logger.debug(f"[{self.udf_signature}] filtered_objects: {filtered_objects}, filtered_subjects: {filtered_subjects}, filtered_targets: {filtered_targets}")
        logger.info(f"[{self.udf_signature}] Model distillation (initialization) finished")

        num_active_learning_rounds = (self.n_train_distill - 1) // 100
        labeled_indices = set()
        for active_learning_round in range(num_active_learning_rounds + 1):
            logger.info(f"[{self.udf_signature}] Active learning round: {active_learning_round}")

            if active_learning_round == 0:
                logger.info(f"[{self.udf_signature}] Model distillation (data loading) started")
                _start = time.time()
                # NOTE: LLM refuses to generate labels in some cases, so we need to double the number of samples (i.e., self.n_train_distill * 1.2) to ensure we have enough training samples
                if self.gt_udf_name:
                    self.df_train, self.df_test = self.construct_train_and_test_data(self.n_obj, int(self.n_train_distill * 1.2), self.n_test_distill, df_with_img_column=True, filtered_objects=filtered_objects, filtered_subjects=filtered_subjects, filtered_targets=filtered_targets)
                else:
                    self.df_train = self.construct_train_and_test_data(self.n_obj, int(self.n_train_distill * 1.2), df_with_img_column=True, filtered_objects=filtered_objects, filtered_subjects=filtered_subjects, filtered_targets=filtered_targets)
                logger.info(f"[{self.udf_signature}] Model distillation (data loading) finished")
                self.execution_time["model_distillation_data_loading"] += time.time() - _start

            logger.info(f"[{self.udf_signature}] Model distillation (data labeling) started")
            await self.llm_annotate_data(active_learning_round=active_learning_round)
            logger.info(f"[{self.udf_signature}] Model distillation (data labeling) finished")

            logger.info(f"[{self.udf_signature}] Model distillation (model training) started")
            _start = time.time()
            self.mlp_prepare_data()
            best_ckpt = self.train(active_learning_round)
            if self.gt_udf_name and hasattr(self, 'df_test'):
                self.test()
            logger.info(f"[{self.udf_signature}] Model distillation (model training) finished")
            self.execution_time["model_distillation_model_training"] += time.time() - _start

            if active_learning_round < num_active_learning_rounds:
                logger.info(f"[{self.udf_signature}] Model distillation (active learning) started")
                _start = time.time()
                checkpoint = torch.load(best_ckpt)
                hyper_parameters = checkpoint["hyper_parameters"]
                best_mlp_model = mlp.MLPProd(**hyper_parameters)
                best_mlp_model.load_state_dict(checkpoint["state_dict"])
                best_mlp_model.eval()
                best_mlp_model.to(self.device)

                # Predict on the training split
                pred_dataset = PredImageDataset(self.conn, self.n_obj, self.attribute_features_dir, self.relationship_features_dir)
                pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=4096, num_workers=self.num_workers, shuffle=False)

                all_rows = []
                all_predictions = []
                all_uncertainties = []

                with torch.no_grad():
                    for row, feature in tqdm(pred_loader, file=sys.stdout):
                        feature = feature.to(self.device)
                        pred, uncertainty = best_mlp_model(feature)
                        all_rows.append(row)
                        all_predictions.append(pred.cpu())
                        all_uncertainties.append(uncertainty.cpu())

                rows = torch.cat(all_rows).tolist()
                predictions = torch.cat(all_predictions).tolist()
                uncertainties = torch.cat(all_uncertainties).tolist()

                if self.n_obj == 1:
                    check_df = pd.DataFrame(rows, columns=['vid', 'fid', 'oid'])
                    check_df['aname'] = self.gt_udf_name
                    result = check_df.merge(attribute_df, on=['vid', 'fid', 'oid', 'aname'], how='left', indicator=True)
                    result = result.drop_duplicates(subset=['vid', 'fid', 'oid', 'aname'])
                    result = result.rename(columns={"oid": "o1_oid"})
                else:
                    check_df = pd.DataFrame(rows, columns=['vid', 'fid', 'oid1', 'oid2'])
                    check_df['rname'] = self.gt_udf_name
                    result = check_df.merge(relationship_df, on=['vid', 'fid', 'oid1', 'oid2', 'rname'], how='left', indicator=True)
                    result = result.drop_duplicates(subset=['vid', 'fid', 'oid1', 'oid2', 'rname'])
                    result = result.rename(columns={"oid1": "o1_oid", "oid2": "o2_oid"})
                result['label'] = (result['_merge'] == 'both').astype(int)
                result = result.reset_index(drop=True)
                labels = result['label'].tolist()

                # Compute F1 score
                f1 = f1_score(labels, predictions)
                logger.info(f"[{self.udf_signature}] F1 score: {f1}")
                tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
                logger.info(f"[{self.udf_signature}] TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
                if self.dataset == "charades":
                    result['pred'] = predictions
                    result['uncertainty'] = uncertainties
                    result_human_object = result[result["o1_oid"] == 0]
                    labels_human_object = result_human_object["label"].tolist()
                    predictions_human_object = result_human_object["pred"].tolist()
                    f1_human_object = f1_score(labels_human_object, predictions_human_object)
                    logger.info(f"[{self.udf_signature}] [human-object only] F1 score: {f1_human_object}")
                    tn_1, fp_1, fn_1, tp_1 = confusion_matrix(labels_human_object, predictions_human_object).ravel()
                    logger.info(f"[{self.udf_signature}] [human-object only] TP: {tp_1}, FP: {fp_1}, TN: {tn_1}, FN: {fn_1}")

                # Active learning: select a batch of rows with the highest uncertainty that are not labeled
                selected_indices = np.argsort(-np.array(uncertainties))
                if self.dataset == "charades":
                    # "charades": only select the rows with the highest uncertainty for the human-object relationship
                    mask = (result['o1_oid'] == 0) & (result['vid'] < self.config[self.dataset]["dataset_size"] // 2)
                    filtered_indices = set(result.index[mask].tolist())
                    selected_indices = [i for i in selected_indices if i in filtered_indices and i not in labeled_indices]
                else:
                    # The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
                    mask = (result['vid'] < self.config[self.dataset]["dataset_size"] // 2)
                    filtered_indices = set(result.index[mask].tolist())
                    selected_indices = [i for i in selected_indices if i in filtered_indices and i not in labeled_indices]
                selected_indices = selected_indices[:min(100, self.n_train_distill - 100 * active_learning_round)]
                # Random sampling:
                # selected_indices = np.random.choice(len(uncertainties), min(100, self.n_train_distill - 100 * active_learning_round), replace=False)
                labeled_indices.update(selected_indices)
                logger.info(f"[{self.udf_signature}] labeled_indices: {sorted(labeled_indices)}, len(labeled_indices): {len(labeled_indices)}")
                if self.n_obj == 1:
                    columns = ['vid', 'fid', 'o1_oid']
                    df_source = self.one_object_df
                else:
                    columns = ['vid', 'fid', 'o1_oid', 'o2_oid']
                    df_source = self.two_objects_df
                selected_rows = result.iloc[selected_indices][columns]
                self.df_train = df_source.merge(selected_rows, on=columns, how='inner').reset_index(drop=True)
                self.df_train = self.df_train.drop_duplicates(subset=columns)
                self.df_train["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, self.df_train["vid"], self.df_train["fid"]), total=len(self.df_train), file=sys.stdout, desc="Processing frames"))
                logger.info(f"[{self.udf_signature}] Model distillation (active learning) finished")
                self.execution_time["model_distillation_active_learning"] += time.time() - _start

        udf_dict = {}
        udf_dict["udf_name"] = self.udf_name
        udf_dict["udf_signature"] = self.udf_signature
        udf_dict["udf_description"] = self.udf_description
        udf_dict["semantic_interpretation"] = 'model'
        udf_dict["function_implementation"] = best_ckpt
        new_udf_candidate = UDFCandidate(id='model', payload=udf_dict)
        logger.info(f"[{self.udf_signature}] Model distillation completed")
        return [new_udf_candidate]

    async def llm_filter_relevant_subjects_targets(self, udf_signature, udf_description):
        res = await self._llm_filter_relevant_training_data(udf_signature, udf_description, prompt_key="filter_subject_target")
        return res["subjects"], res["targets"]

    async def llm_filter_relevant_objects(self, udf_signature, udf_description):
        res = await self._llm_filter_relevant_training_data(udf_signature, udf_description, prompt_key="filter_object")
        return res["answer"]

    async def _llm_filter_relevant_training_data(self, udf_signature, udf_description, prompt_key):
        filter_objects_prompt = replace_slot(
            self.prompt_config[prompt_key],
            {
                "object_classes": self.object_domain,
                "udf_signature": udf_signature,
                "udf_description": udf_description,
            },
        )
        logger.debug(f"[{self.udf_signature}] filter_objects_prompt: {filter_objects_prompt}")
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                logger.debug(f"[{self.udf_signature}] trial: {trial}")
                response = await self.completion_with_backoff(
                    model=self.openai_model_name,
                    messages=[
                        {"role": "user", "content": filter_objects_prompt},
                    ],
                    temperature=self.config["udf_generator"]["temperature"],
                    top_p=self.config["udf_generator"]["top_p"],
                    seed=self.run_id * 42 + trial,
                )
                self.cost_estimation["filter_relevant_objects"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
                res = eval(
                    "\n\n".join(
                        re.findall(
                            r"```json\n(.*?)```",
                            response.choices[0].message.content,
                            re.DOTALL,
                        )
                    )
                )
                return res
            except Exception as e:
                logger.exception(f"[{self.udf_signature}] ERROR: failed to filter relevant objects: {e}")
                logger.debug(f"[{self.udf_signature}] {response}")

    async def llm_annotate_data(self, batch_size=8, active_learning_round=0):
        _start = time.time()
        labeled_data = defaultdict(list) # dictionary with 'train' and 'test' fields. Each field is a list of tuples (image_features, label)
        llm_positive_df = []
        llm_negative_df = []
        labeled_data_dir = os.path.join(self.config["model_dir"], "labeled_data", self.dataset, self.llm_method, self.query_class_name)
        labeled_data_path = os.path.join(labeled_data_dir, "udf-{}_query-{}_run-{}_ntrain-{}_labeled_data.pt".format(self.udf_name.lower(), self.query_id, self.run_id, self.n_train_distill))
        os.makedirs(labeled_data_dir, exist_ok=True)
        if self.load_labeled_data and os.path.exists(labeled_data_path):
            logger.info(f"[{self.udf_signature}] Loading labeled data from {labeled_data_path}")
            labeled_data = torch.load(labeled_data_path)
            for data in labeled_data['train']:
                logger.debug("[{}] row: {}".format(self.udf_signature, data['row'].drop('img').to_dict()))
                logger.debug("[{}] base64_image: {}".format(self.udf_signature, data["base64_image"]))
                logger.debug("[{}] gt_label: {}, llm_label: {}".format(self.udf_signature, data["label"], data["llm_label"]))
                if data["llm_label"] == 1:
                    llm_positive_df.append(data['row'])
                else:
                    llm_negative_df.append(data['row'])
            logger.debug("[{}] llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(self.udf_signature, labeled_data["metadata"]["llm_TP"], labeled_data["metadata"]["llm_FP"], labeled_data["metadata"]["llm_TN"], labeled_data["metadata"]["llm_FN"], labeled_data["metadata"]["llm_f1"]))
            if "test" in labeled_data:
                logger.debug("[{}] test_pos: {}, test_neg: {}".format(self.udf_signature, labeled_data["metadata"]["test_pos"], labeled_data["metadata"]["test_neg"]))
        else:
            self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN = 0, 0, 0, 0
            # Training and validation data
            self.label_count = 0
            if self.llm_method == "gpt4v":
                if self.is_async:
                    self.execution_time["model_distillation_data_labeling"] += time.time() - _start
                    tasks = [asyncio.create_task(self.label_one(row, labeled_data, llm_positive_df, llm_negative_df, active_learning_round)) for _, row in self.df_train.iterrows()]
                    try:
                        await asyncio.gather(*tasks)
                    except asyncio.CancelledError:
                        # Raised when we collected enough valid results. Cancel all other tasks
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                    _start = time.time()
                else:
                    for _, row in self.df_train.iterrows():
                        logger.debug(f"[{self.udf_signature}] +++++++++++++++++++++++++++++++++++++++++++++++")
                        try:
                            gt_label = self._get_gt_label(row)
                            # Read and crop frame
                            logger.debug("[{}] row: {}".format(self.udf_signature, row.drop('img').to_dict()))
                            frame, image_size = self.frame_processing_for_model(row)
                            if frame is None:
                                continue
                            self.execution_time["model_distillation_data_labeling"] += time.time() - _start
                            llm_label, base64_image, image_prompt = await self._llm_annotate_frame(frame, image_size, row, gt_label)
                            _start = time.time()
                            labeled_data['train'].append({"label": gt_label, "llm_label": llm_label, "base64_image": base64_image, "image_prompt": image_prompt, "row": row})
                            if llm_label == 1:
                                llm_positive_df.append(row)
                            else:
                                llm_negative_df.append(row)
                            if self.label_count >= min(100, self.n_train_distill - 100 * active_learning_round):
                                break
                        except Exception as e:
                            logger.exception(f"[{self.udf_signature}] Error: {e}")
                            continue
            elif self.llm_method == "llava":
                batched_rows = []
                batched_frames = []
                batched_image_sizes = []
                batched_gt_labels = []
                for _, row in self.df_train.iterrows():
                    try:
                        # Read and crop frame
                        frame, image_size = self.frame_processing_for_model(row)
                        if frame is None:
                            continue
                        gt_label = self._get_gt_label(row)
                        batched_rows.append(row)
                        batched_frames.append(frame)
                        batched_image_sizes.append(image_size)
                        batched_gt_labels.append(gt_label)
                        if len(batched_rows) == batch_size:
                            generated_text, llm_labels, base64_images, image_prompts = self._llava_annotate_frame(batched_frames, batched_image_sizes, batched_rows, batched_gt_labels)
                            for i in range(len(batched_rows)):
                                if llm_labels[i] == -1:
                                    continue
                                logger.debug(f"[{self.udf_signature}] +++++++++++++++++++++++++++++++++++++++++++++++")
                                logger.debug("[{}] row: {}".format(self.udf_signature, batched_rows[i].drop('img').to_dict()))
                                logger.debug("[{}] base64_image: {}".format(self.udf_signature, base64_images[i]))
                                logger.debug("[{}] Llava image prompt: {}".format(self.udf_signature, image_prompts[i]))
                                logger.debug("[{}] Llava result: {}".format(self.udf_signature, generated_text[i]))
                                logger.debug("[{}] gt_label: {}".format(self.udf_signature, batched_gt_labels[i]))
                                labeled_data['train'].append({"label": batched_gt_labels[i], "llm_label": llm_labels[i], "base64_image": base64_images[i], "image_prompt": image_prompts[i], "row": batched_rows[i]})
                                if llm_labels[i] == 1:
                                    llm_positive_df.append(batched_rows[i])
                                else:
                                    llm_negative_df.append(batched_rows[i])
                            if self.label_count >= self.n_train_distill:
                                break
                            batched_rows = []
                            batched_frames = []
                            batched_image_sizes = []
                            batched_gt_labels = []
                    except Exception as e:
                        logger.exception(f"[{self.udf_signature}] Error: {e}")
                        continue
            elif self.llm_method == "user": # Ground truth labels
                for _, row in self.df_train.iterrows():
                    logger.debug(f"[{self.udf_signature}] +++++++++++++++++++++++++++++++++++++++++++++++")
                    try:
                        gt_label = self._get_gt_label(row)
                        # Read and crop frame
                        logger.debug("[{}] row: {}".format(self.udf_signature, row.drop('img').to_dict()))
                        # frame, image_size = self.frame_processing_for_model(row)
                        # if self.n_obj == 2:
                        #     o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
                        #     cv2.rectangle(frame, (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
                        #     cv2.rectangle(frame, (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(255, 0, 0), thickness=1)
                        # _, buffer = cv2.imencode('.jpg', frame)
                        # base64_image = base64.b64encode(buffer).decode('utf-8')
                        # logger.debug("base64_image: {}".format(base64_image))
                        logger.debug("[{}] gt_label: {}".format(self.udf_signature, gt_label))
                        if gt_label == 1:
                            self.llm_TP += 1
                        else:
                            self.llm_TN += 1
                        self.label_count += 1
                        labeled_data['train'].append({"label": gt_label, "llm_label": gt_label, "base64_image": "", "image_prompt": "", "row": row})
                        if gt_label == 1:
                            llm_positive_df.append(row)
                        else:
                            llm_negative_df.append(row)
                        if self.label_count >= self.n_train_distill:
                            break
                    except Exception as e:
                        logger.exception(f"[{self.udf_signature}] Error: {e}")
                        continue
            if self.gt_udf_name is not None:
                llm_f1 = 2*self.llm_TP/(2*self.llm_TP+self.llm_FP+self.llm_FN) if 2*self.llm_TP+self.llm_FP+self.llm_FN > 0 else 0.0
                logger.debug("[{}] llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(self.udf_signature, self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN, llm_f1))
            else:
                llm_f1 = -1
            labeled_data["metadata"] = {"llm_TP": self.llm_TP, "llm_FP": self.llm_FP, "llm_TN": self.llm_TN, "llm_FN": self.llm_FN, "llm_f1": llm_f1}

            llm_positive_df = pd.DataFrame(llm_positive_df).reset_index(drop=True)
            self.llm_positive_df = pd.concat([self.llm_positive_df, llm_positive_df], ignore_index=True) if self.llm_positive_df is not None else llm_positive_df
            llm_negative_df = pd.DataFrame(llm_negative_df).reset_index(drop=True)
            self.llm_negative_df = pd.concat([self.llm_negative_df, llm_negative_df], ignore_index=True) if self.llm_negative_df is not None else llm_negative_df

            # Test data
            if active_learning_round == 0 and self.gt_udf_name is not None and hasattr(self, 'df_test'):
                for _, row in self.df_test.iterrows():
                    try:
                        gt_label = self._get_gt_label(row)
                        labeled_data['test'].append({"label": gt_label, "row": row})
                    except Exception as e:
                        logger.exception(f"[{self.udf_signature}] Error: {e}")
                        continue
                pos_count = sum([1 for data in labeled_data['test'] if data["label"] == 1])
                neg_count = sum([1 for data in labeled_data['test'] if data["label"] == 0])
                logger.debug(f"[{self.udf_signature}] test_pos: {pos_count}, test_neg: {neg_count}")
                labeled_data["metadata"]["test_pos"] = pos_count
                labeled_data["metadata"]["test_neg"] = neg_count
            # save labeled_data to a file
            if self.save_labeled_data:
                logger.info(f"[{self.udf_signature}] Saving labeled data to {labeled_data_path}")
                torch.save(labeled_data, labeled_data_path)
        if active_learning_round == 0:
            self.labeled_data = labeled_data
        else:
            self.labeled_data['train'].extend(labeled_data['train'])
        self.execution_time["model_distillation_data_labeling"] += time.time() - _start
        # if self.llm_method == "llava":
        #     del llava_model
        #     torch.cuda.empty_cache()

    def mlp_prepare_data(self):
        splits = ['train', 'test'] if self.gt_udf_name is not None else ['train']
        for split in splits:
            logger.info(f"[{self.udf_signature}] Processing {split} data")
            idx_to_remove = []
            for i in range(len(self.labeled_data[split])):
                if "image_features" in self.labeled_data[split][i]:
                    continue
                try:
                    data = self.labeled_data[split][i]
                    row = data['row']
                    frame, image_size = self.frame_processing_for_model(row)
                    if frame is None: # failed to read the frame
                        idx_to_remove.append(i)
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_features = self.extract_features(frame, row, image_size)
                    if self.dataset == "charades" and self.n_obj == 2:
                        # For charades, also include object class embeddings
                        text_features = self.extract_text_features(row)
                        self.labeled_data[split][i]["image_features"] = torch.cat([image_features, text_features], dim=-1)
                    else:
                        self.labeled_data[split][i]["image_features"] = image_features
                except Exception as e:
                    logger.exception(f"[{self.udf_signature}] Error: {e}")
                    idx_to_remove.append(i)
                    continue
            for i in reversed(idx_to_remove):
                del self.labeled_data[split][i]

        # use 20% of the training data as validation data
        self.train_dataset = CustomImageDataset(self.labeled_data['train'], train=True)
        if self.gt_udf_name is not None:
            test_dataset = CustomImageDataset(self.labeled_data['test'], train=False)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    def train(self, active_learning_round=-1):
        # logger.debug("mlp_config: {}".format(mlp_config))
        if self.dataset == "charades":
            mlp_dim_in = self.dim_in if self.n_obj == 1 else self.dim_in * 5
        else:
            mlp_dim_in = self.dim_in if self.n_obj == 1 else self.dim_in * 3
        logger.debug(f"[{self.udf_signature}] mlp_dim_in: {mlp_dim_in}") # should be 512 for clip-vit-base-patch32
        self.checkpoint_root = os.path.join(self.config["model_dir"], "model_udf", self.dataset, f"{self.llm_method}_{self.mlp_method}", self.udf_name.lower())
        if active_learning_round >= 0:
            self.checkpoint_root = os.path.join(self.checkpoint_root, f"active_learning_round_{active_learning_round}")

        # TODO: fine-tuning the learning rate
        # learningrate_callback = pl.callbacks.LearningRateFinder()
        # callbacks.append(learningrate_callback)

        best_model_score = float('inf')
        best_ckpt = None

        class_counts = [sum(data["llm_label"] == i for data in self.labeled_data['train']) for i in range(2)]
        try:
            self.class_weights = torch.tensor([1.0 / count for count in class_counts]).to(self.device)
        except ZeroDivisionError as e:
            logger.exception(f"[{self.udf_signature}] Error: {e}\nclass_counts: {class_counts}")
            self.class_weights = torch.tensor([1.0, 1.0]).to(self.device)

        logger.debug(f"[{self.udf_signature}] class_counts: {class_counts}, class_weights: {self.class_weights}")

        for i in range(10):
            logger.debug(f"[{self.udf_signature}] Training model: trial {i}")
            self.checkpoint_filename = "udf={}-run={}-ntrain={}-trial={}".format(self.udf_name.lower(), self.run_id, self.n_train_distill, i)
            os.makedirs(self.checkpoint_root, exist_ok=True)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filename=self.checkpoint_filename,
                monitor="val_loss",
                mode="min",
            )
            callbacks=[checkpoint_callback]
            earlystopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
            callbacks.append(earlystopping_callback)
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)

            train_set_size = int(len(self.train_dataset) * 0.8)
            valid_set_size = len(self.train_dataset) - train_set_size
            train_split, valid_split = torch.utils.data.random_split(self.train_dataset, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(self.run_id * 42 + i))

            self.train_loader = torch.utils.data.DataLoader(train_split, batch_size=512, shuffle=True)
            self.val_loader = torch.utils.data.DataLoader(valid_split, batch_size=512, shuffle=False)

            self.mlp_model = mlp.MLP(mlp_dim_in, 2, logger, self.class_weights) # binary classification

            self.trainer = pl.Trainer(
                # deterministic=self.deterministic,
                max_epochs=50,
                devices=1,
                accelerator="auto",
                enable_progress_bar=True,
                enable_checkpointing=True,
                enable_model_summary=False,
                # logger=pl_logger,
                default_root_dir=self.checkpoint_root,
                callbacks=callbacks,
                # check_val_every_n_epoch=5,
                # log_every_n_steps=min(50, len(dataset)-1),
                log_every_n_steps=1
            )

            self.trainer.fit(
                self.mlp_model,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.val_loader
            )

            # retrieve the best checkpoint after training
            current_model_score = checkpoint_callback.best_model_score
            logger.debug(f"[{self.udf_signature}] current_model_score: {current_model_score}, best_model_score: {min(best_model_score, current_model_score)}")
            if current_model_score < best_model_score:
                best_model_score = current_model_score
                best_ckpt = checkpoint_callback.best_model_path
        logger.debug(f"[{self.udf_signature}] Best model checkpoint: {best_ckpt}")
        # best_mlp_model = mlp.MLP.load_from_checkpoint(best_ckpt)
        return best_ckpt

    def test(self):
        # Inference:
        # model = mlp.MLP.load_from_checkpoint(self.dim_in, 2)
        logger.debug(f"[{self.udf_signature}] test with last model: ")
        self.trainer.test(self.mlp_model, dataloaders=self.test_loader)
        logger.debug(f"[{self.udf_signature}] test with best model: ")
        self.trainer.test(ckpt_path="best", dataloaders=self.test_loader)

    def _get_gt_label(self, row):
        if self.gt_udf_name is None:
            return None
        if self.n_obj == 1:
            return int(self.gt_udf_name in row["o1_gt_anames"])
        else:
            return int(self.gt_udf_name in row["o1_o2_gt_rnames"])

    def frame_processing_for_model(self, row):
        frame = row['img']
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_size = frame.shape[:2]
        if self.n_obj == 1:
            factor = 1 if self.dataset == "cityflow" else 1.5
            x1, y1, x2, y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size, factor=factor)
            frame = frame[y1:y2, x1:x2]
        else:
            o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
            o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
            frame = frame[min(o1_y1, o2_y1):max(o1_y2, o2_y2), min(o1_x1, o2_x1):max(o1_x2, o2_x2)]
        # resize the frame to 224x224
        frame = cv2.resize(frame, (224, 224))
        return frame, image_size


    async def label_one(self, row, labeled_data, llm_positive_df, llm_negative_df, active_learning_round):
        log_msgs = []
        log_msgs.append(f"[{self.udf_signature}] +++++++++++++++++++++++++++++++++++++++++++++++")
        # logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++")

        try:
            gt_label = self._get_gt_label(row)
            # Read and crop frame
            log_msgs.append("[{}] row: {}".format(self.udf_signature, row.drop('img').to_dict()))
            # logger.debug("row: {}".format(row.drop('img').to_dict()))
            frame, image_size = self.frame_processing_for_model(row)
            if frame is None:
                logger.debug("\n".join(log_msgs))
                return
            llm_label, base64_image, image_prompt = await self._async_llm_annotate_frame(frame, image_size, row, gt_label, log_msgs)
        except Exception as e:
            logger.debug("\n".join(log_msgs))
            logger.exception(f"[{self.udf_signature}] Error: {e}")
            return

        if len(labeled_data['train']) >= min(100, self.n_train_distill - 100 * active_learning_round):
            raise asyncio.CancelledError

        logger.debug("\n".join(log_msgs))
        labeled_data['train'].append({"label": gt_label, "llm_label": llm_label, "base64_image": base64_image, "image_prompt": image_prompt, "row": row})
        if llm_label == 1:
            llm_positive_df.append(row)
        else:
            llm_negative_df.append(row)


    async def _async_llm_annotate_frame(self, frame, image_size, row, gt_label, log_msgs):
        # TODO: don't resize the frame here, resize it in the model
        # Convert the frame to a base 64 encoded image
        # TODO: Try different llm annotation prompt: draw bounding boxes on subject and object.
        if self.n_obj == 2:
            o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
            cv2.rectangle(frame, (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
            cv2.rectangle(frame, (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(255, 0, 0), thickness=1)
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        log_msgs.append(f"[{self.udf_signature}] base64_image: {base64_image}")
        # logger.debug("base64_image: {}".format(base64_image))
        image_prompt = self._create_image_prompt(row, image_size)
        log_msgs.append(f"[{self.udf_signature}] Image prompt: {image_prompt}")
        # logger.debug("Image prompt: {}".format(image_prompt))
        response = await self.completion_with_backoff(
            model=self.openai_model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": image_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=10,
            temperature=0.2,
            top_p=0.5,
            seed=self.run_id
        )
        result = response.choices[0].message.content
        self.cost_estimation["model_udf_data_labeling"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
        log_msgs.append(f"[{self.udf_signature}] Result: {result}")
        # logger.debug("Result: {}".format(result))
        log_msgs.append(f"[{self.udf_signature}] gt_label: {gt_label}")
        # logger.debug("gt_label: {}".format(gt_label))
        if "yes" in result.lower():
            llm_label = 1
            if self.gt_udf_name is not None:
                if gt_label == 1:
                    self.llm_TP += 1
                else:
                    self.llm_FP += 1
        elif "no" in result.lower():
            llm_label = 0
            if self.gt_udf_name is not None:
                if gt_label == 0:
                    self.llm_TN += 1
                else:
                    self.llm_FN += 1
        else:
            raise ValueError("Invalid response", result)
        self.label_count += 1
        return llm_label, base64_image, image_prompt

    async def _llm_annotate_frame(self, frame, image_size, row, gt_label):
        # TODO: don't resize the frame here, resize it in the model
        # Convert the frame to a base 64 encoded image
        # TODO: Try different llm annotation prompt: draw bounding boxes on subject and object.
        _start = time.time()
        if self.n_obj == 2:
            o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
            cv2.rectangle(frame, (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
            cv2.rectangle(frame, (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(255, 0, 0), thickness=1)
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        logger.debug(f"[{self.udf_signature}] base64_image: {base64_image}")
        image_prompt = self._create_image_prompt(row, image_size)
        logger.debug(f"[{self.udf_signature}] Image prompt: {image_prompt}")
        self.execution_time["model_distillation_data_labeling"] += time.time() - _start
        response = await self.completion_with_backoff(
            model=self.openai_model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": image_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=10,
            temperature=0.2,
            top_p=0.5,
            seed=self.run_id
        )
        _start = time.time()
        result = response.choices[0].message.content
        self.cost_estimation["model_udf_data_labeling"] += response.usage.prompt_tokens * MODEL_COST[self.openai_model_name][0] + response.usage.completion_tokens * MODEL_COST[self.openai_model_name][1]
        logger.debug(f"[{self.udf_signature}] Result: {result}")
        logger.debug(f"[{self.udf_signature}] gt_label: {gt_label}")
        if "yes" in result.lower():
            llm_label = 1
            if self.gt_udf_name is not None:
                if gt_label == 1:
                    self.llm_TP += 1
                else:
                    self.llm_FP += 1
        elif "no" in result.lower():
            llm_label = 0
            if self.gt_udf_name is not None:
                if gt_label == 0:
                    self.llm_TN += 1
                else:
                    self.llm_FN += 1
        else:
            self.execution_time["model_distillation_data_labeling"] += time.time() - _start
            raise ValueError("Invalid response", result)
        self.label_count += 1
        self.execution_time["model_distillation_data_labeling"] += time.time() - _start
        return llm_label, base64_image, image_prompt

    def _create_image_prompt(self, row, image_size):
        image_prompt = "{}? Answer with 'yes' or 'no'.".format(self.replace_objects(self.udf_description.rstrip(string.punctuation), row, image_size))
        return image_prompt

    def _llava_annotate_frame(self, batched_frames, batched_image_sizes, batched_rows, batched_gt_labels):
        llm_labels, base64_images, llava_image_prompts = [], [], []
        pil_images = []
        for i in range(len(batched_frames)):
            if self.n_obj == 2:
                # Draw bounding boxes on subject and object
                o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(batched_rows[i], batched_image_sizes[i])
                cv2.rectangle(batched_frames[i], (int(o1x1), int(o1y1)), (int(o1x2), int(o1y2)), color=(0, 0, 255), thickness=1)
                cv2.rectangle(batched_frames[i], (int(o2x1), int(o2y1)), (int(o2x2), int(o2y2)), color=(255, 0, 0), thickness=1)
            _, buffer = cv2.imencode('.jpg', batched_frames[i])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            base64_images.append(base64_image)
            batched_frames[i] = cv2.cvtColor(batched_frames[i], cv2.COLOR_BGR2RGB)
            # Convert the frame to PIL image
            pil_image = Image.fromarray(batched_frames[i])
            pil_images.append(pil_image)
            llava_image_prompt, last_word = self._create_llava_image_prompt(batched_rows[i], batched_image_sizes[i])
            llava_image_prompts.append(llava_image_prompt)
        inputs = self.llava_processor(llava_image_prompts, pil_images, padding=True, return_tensors="pt").to(self.device)
        output = self.llava_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.2,
            top_p=0.7
        )
        generated_text = self.llava_processor.batch_decode(output, skip_special_tokens=True)
        results = []
        for i, result in enumerate(generated_text):
            result = result.split(last_word)[-1].strip().lower()
            results.append(result)
            if "yes" in result.lower():
                llm_label = 1
                if self.gt_udf_name is not None:
                    if batched_gt_labels[i] == 1:
                        self.llm_TP += 1
                    else:
                        self.llm_FP += 1
                self.label_count += 1
            elif "no" in result.lower():
                llm_label = 0
                if self.gt_udf_name is not None:
                    if batched_gt_labels[i] == 0:
                        self.llm_TN += 1
                    else:
                        self.llm_FN += 1
                self.label_count += 1
            else:
                llm_label = -1
            llm_labels.append(llm_label)
        return results, llm_labels, base64_images, llava_image_prompts

    def _create_llava_image_prompt(self, row, image_size):
        image_prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{}? Answer with 'yes' or 'no'.<|im_end|><|im_start|>assistant\n".format(self.replace_objects(self.udf_description.rstrip(string.punctuation), row, image_size))
        return image_prompt, "assistant"

    def replace_objects(self, input_string, row, image_size):
        # Find all occurrences of "o" followed by integers
        objects = re.findall(r'o\d+', input_string)
        # Sort the objects based on the integer part of the identifier

        h, w = image_size
        if len(objects) == 1:
            new_string = input_string.replace(objects[0], f"{row['o1_oname']} {objects[0]}")
        elif len(objects) == 2 and self.n_obj == 2:
            o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
            new_string = input_string.replace(objects[0], f"{row['o1_oname']} {objects[0]} at {int(o1x1), int(o1y1), int(o1x2), int(o1y2)} in the red box")
            new_string = new_string.replace(objects[1], f"{row['o2_oname']} {objects[1]} at {int(o2x1), int(o2y1), int(o2x2), int(o2y2)} in the blue box")
        else:
            new_string = input_string

        return new_string

    def extract_features(self, frame, row, image_size):
        """
        three CLIP features: original image, subject mask, target mask
        """
        if self.n_obj == 1:
            inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            outputs = outputs.squeeze(0)
        else:
            o1x1, o1y1, o1x2, o1y2, o2x1, o2y1, o2x2, o2y2 = self._compute_new_box_after_crop(row, image_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_subject = frame.copy()
            frame_target = frame.copy()
            # set the pixels outside the bounding box to 0
            frame_subject[:int(o1y1), :] = 0
            frame_subject[int(o1y2):, :] = 0
            frame_subject[:, :int(o1x1)] = 0
            frame_subject[:, int(o1x2):] = 0
            frame_target[:int(o2y1), :] = 0
            frame_target[int(o2y2):, :] = 0
            frame_target[:, :int(o2x1)] = 0
            frame_target[:, int(o2x2):] = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_subject = cv2.cvtColor(frame_subject, cv2.COLOR_BGR2RGB)
            frame_target = cv2.cvtColor(frame_target, cv2.COLOR_BGR2RGB)
            inputs = self.clip_processor(images=[frame, frame_subject, frame_target], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            outputs = outputs.reshape(-1)
        return outputs

    def extract_text_features(self, row):
        if self.n_obj == 1:
            text = row["o1_oname"]
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**inputs)
            outputs = outputs.squeeze(0)
        else:
            inputs = self.tokenizer([row["o1_oname"], row["o2_oname"]], padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**inputs) # 2 x 512
            outputs = outputs.reshape(-1)

        return outputs
