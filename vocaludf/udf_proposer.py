from typing_extensions import Annotated
import autogen
import json
from typing import Tuple, List
import os
import math
from vocaludf.utils import (
    replace_slot,
    parse_signature,
    transform_function,
    PredImageDataset
)
from vocaludf.pretrained_model_api import image_captioning, image_classification, visual_question_answering, object_detection, depth_estimation
import time
import resource
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
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch
from torch.utils.data import Dataset
import base64
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import string
import lightning.pytorch as pl
from vocaludf import mlp

tqdm.pandas()
client = OpenAI()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

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

class UDFCandidate:
    def __init__(self, id, payload):
        self.kwargs = payload.get("kwargs", {})
        self.id = str(id) + "_" + "_".join([f"{k}-{v}" for k, v in self.kwargs.items()]) if self.kwargs else str(id) # 'model' for model-based UDFs
        self.udf_name = payload["udf_name"]
        self.udf_signature = payload["udf_signature"]
        self.udf_description = payload["udf_description"]
        self.semantic_interpretation = payload["semantic_interpretation"] # 'model' for model-based UDFs
        self.function_implementation = payload["function_implementation"] # python function for program-based UDFs, and path_to_best_ckpt for model-based UDFs
        self.score = 1  # F1 score
        self.test_score = -1
        self.loss_t = 0  # loss_t = n_misclassified

    def __str__(self):
        return f"UDFCandidate(id: {self.id}, function_implementation: {self.function_implementation}, test_score: {self.test_score}, score: {self.score}, loss_t: {self.loss_t})"


class UDFProposer:
    llm_method = "gpt4v"
    mlp_method = "three_clip"
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
        program_with_pixels,
        query_id,
        run_id,
        num_workers,
        save_generated_udf,
        save_labeled_data,
        load_labeled_data,
        n_train_distill,
        selection_strategy,
        selection_labels,
        allow_kwargs_in_udf,
        save_udf_base_dir,
    ):
        self.config = config
        self.prompt_config = prompt_config
        self.config_list = config_list
        self.registered_functions = registered_functions
        self.dataset = dataset
        self.labeling_budget = labeling_budget
        self.num_interpretations = num_interpretations
        self.num_parameter_search = num_parameter_search
        self.program_with_pixels = program_with_pixels
        self.query_id = query_id
        self.run_id = run_id
        self.num_workers = num_workers
        self.save_generated_udf = save_generated_udf
        self.save_udf_base_dir = save_udf_base_dir
        self.selection_strategy = selection_strategy
        self.selection_labels = selection_labels
        self.allow_kwargs_in_udf = allow_kwargs_in_udf

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.set_active_domain()

        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )

        self.init_table()

        # Create a train and test split
        # NOTE: probably put these values in the config file
        # self.n_train = 10000
        # self.n_test = 10000
        self.n_train = 1000
        self.n_test = 1000

        # Initialization for model distillation
        self.n_train_distill = n_train_distill
        self.n_test_distill = 1000
        self.save_labeled_data = save_labeled_data
        self.load_labeled_data = load_labeled_data
        self.attribute_features_dir = os.path.join(self.config[self.dataset]["features_dir"], "attribute")
        self.relationship_features_dir = os.path.join(self.config[self.dataset]["features_dir"], "relationship")
        # Load the CLIP model
        # clip_model_name = "openai/clip-vit-base-patch32"
        clip_model_name = os.path.join(self.config['model_dir'], 'clip-vit-base-patch32')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        # self.clip_model.save_pretrained(os.path.join(self.config['model_dir'], 'clip-vit-base-patch32'))
        # self.clip_processor.save_pretrained(os.path.join(self.config['model_dir'], 'clip-vit-base-patch32'))

        self.dim_in = self.clip_model.config.projection_dim

    def set_active_domain(self):
        object_domain = self.config[self.dataset]['onames']
        relationship_domain = []
        attribute_domain = []

        for registered_function in self.registered_functions:
            signature = registered_function["signature"]
            registered_function_name, registered_function_vars = parse_signature(signature)
            if len(registered_function_vars) == 2:
                # Relationship UDF
                relationship_domain.append(registered_function_name.lower())
            else:
                # Attribute UDF
                attribute_domain.append(registered_function_name.lower())
        self.object_domain = object_domain
        self.relationship_domain = relationship_domain
        self.attribute_domain = attribute_domain

    def get_active_domain(self):
        return self.object_domain, self.relationship_domain, self.attribute_domain

    def init_table(self):
        # TODO: use object_domain to filter out objects that are not in the domain
        attr_parameters = ','.join('?' for _ in self.attribute_domain)
        sql = f"""
            CREATE TEMPORARY TABLE one_object ON COMMIT DROP AS
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
            CREATE TEMPORARY TABLE two_objects ON COMMIT DROP AS
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


    ##########################################
    ############                 #############
    ############# Proposing UDFs #############
    ############                 #############
    ##########################################
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
        # TODO: incorporate labels (maybe in the selection stage)
        if self.selection_strategy == "program":
            if self.selection_labels == "none" and self.allow_kwargs_in_udf:
                self.allow_kwargs_in_udf = False
                logger.info("Turning off allow_kwargs_in_udf since no labels are provided")
            return self._generate_program(udf_signature, udf_description)
        elif self.selection_strategy == "model":
            return self._distill_model(udf_signature, udf_description)
        elif self.selection_strategy == "llm":
            if self.selection_labels == "none" and self.allow_kwargs_in_udf:
                self.allow_kwargs_in_udf = False
                logger.info("Turning off allow_kwargs_in_udf since no labels are provided")
            llm_decision = self._llm_decides_udf_type(udf_signature, udf_description)
            if llm_decision == "programUDF":
                return self._generate_program(udf_signature, udf_description)
            elif llm_decision == "modelUDF":
                return self._distill_model(udf_signature, udf_description)
            else:
                raise NotImplementedError(f"llm_decision: {llm_decision} is not supported yet.")
        elif self.selection_strategy == "both":
            program_udf_candidates = self._generate_program(udf_signature, udf_description)
            model_udf_candidates = self._distill_model(udf_signature, udf_description)
            return program_udf_candidates + model_udf_candidates


    def _llm_decides_udf_type(self, udf_signature, udf_description):
        decide_udf_type_dict = self.prompt_config["decide_udf_type"]

        if self.program_with_pixels:
            decide_udf_type_base_prompt = f'{decide_udf_type_dict["overall"]} {decide_udf_type_dict["code_with_pixels"]} {decide_udf_type_dict["model"]} {decide_udf_type_dict["output"]}'
        else:
            decide_udf_type_base_prompt = f'{decide_udf_type_dict["overall"]} {decide_udf_type_dict["code_without_pixels"]} {decide_udf_type_dict["model"]} {decide_udf_type_dict["output"]}'

        decide_udf_type_prompt = replace_slot(
            decide_udf_type_base_prompt,
            {
                "udf_description": udf_description,
                "available_concepts": self.object_domain + self.relationship_domain + self.attribute_domain,
            },
        )
        logger.debug("decide_udf_type_prompt: {}".format(decide_udf_type_prompt))
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                logger.debug(f"trial: {trial}")
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-2024-04-09",
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
                logger.debug(f"llm_decision: {llm_decision}")
                if "programUDF" in llm_decision:
                    return "programUDF"
                elif "modelUDF" in llm_decision:
                    return "modelUDF"
            except Exception as e:
                logger.debug("ERROR: failed to decide UDF type: {}".format(e))
                logger.debug(response)

    ##############################################################
    #############                                    #############
    ############# UDF implementation (program-based) #############
    #############                                    #############
    ##############################################################
    def _generate_program(self, udf_signature, udf_description):
        """
        Implements the UDF program based on the given UDF signature and description.

        Args:
            udf_signature (str): The signature of the UDF.
            udf_description (str): The description of the UDF.

        Returns:
            list: A list of UDFCandidate objects representing the implemented UDFs.
        """
        udf_name, udf_vars = parse_signature(udf_signature)
        self.save_udf_dir = os.path.join(self.save_udf_base_dir, udf_name)
        os.makedirs(self.save_udf_dir, exist_ok=True)
        # Step 3: generate semantic interpretations and implement the UDF. Results are saved to disk
        generate_udfs_dict = self.prompt_config["generate_udfs"]
        n_obj = len(udf_vars)
        attr_or_rel = "attribute" if n_obj == 1 else "relationship"
        if self.allow_kwargs_in_udf:
            if self.program_with_pixels:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_with_optional_kwargs"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_pixels_and_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation_with_optional_kwargs"]}'
            else:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_with_optional_kwargs"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_optional_kwargs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation_with_optional_kwargs"]}'
        else:
            if self.program_with_pixels:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs_with_pixels"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs_with_pixels"][attr_or_rel]} {generate_udfs_dict["pretrained_model_list"]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation"]}'
            else:
                generate_udfs_base_prompt = f'{generate_udfs_dict["overall"]} {generate_udfs_dict["task"]} {generate_udfs_dict["details"]} {generate_udfs_dict["inputs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output"]}'
                verify_syntax_correctness_base_prompt = f'{generate_udfs_dict["task"]} {generate_udfs_dict["semantic_interpretation"]} {generate_udfs_dict["inputs"][attr_or_rel]} {generate_udfs_dict["comments"]} {generate_udfs_dict["output_one_implementation"]}'
        # Construct python function arguments
        if n_obj == 1:
            py_func_args = [f"{udf_vars[0]}_oname", f"{udf_vars[0]}_x1", f"{udf_vars[0]}_y1", f"{udf_vars[0]}_x2", f"{udf_vars[0]}_y2", f"{udf_vars[0]}_anames", "height", "width"]
        else:
            py_func_args = [f"{udf_vars[0]}_oname", f"{udf_vars[0]}_x1", f"{udf_vars[0]}_y1", f"{udf_vars[0]}_x2", f"{udf_vars[0]}_y2", f"{udf_vars[0]}_anames", f"{udf_vars[1]}_oname", f"{udf_vars[1]}_x1", f"{udf_vars[1]}_y1", f"{udf_vars[1]}_x2", f"{udf_vars[1]}_y2", f"{udf_vars[1]}_anames", f"{udf_vars[0]}_{udf_vars[1]}_rnames", f"{udf_vars[1]}_{udf_vars[0]}_rnames", "height", "width"]
        if self.program_with_pixels:
            py_func_args.insert(0, "img")
        py_func_signature = f"{udf_name}({', '.join(py_func_args)})"
        logger.info(
            f"Implementing UDF: {py_func_signature}, with {self.num_interpretations} semantic interpretations"
        )
        generate_udfs_prompt = replace_slot(
            generate_udfs_base_prompt,
            {
                "num_interpretations": self.num_interpretations,
                "udf_signature": py_func_signature,
                "udf_description": udf_description,
                "object_domain": self.object_domain,
                "relationship_domain": self.relationship_domain,
                "attribute_domain": self.attribute_domain,
                "o1": udf_vars[0],
                "o2": udf_vars[1] if n_obj == 2 else "",
                # "n_obj": "one object" if n_obj == 1 else "two objects",
            },
        )
        logger.debug("generate_udfs_prompt: {}".format(generate_udfs_prompt))
        for trial in range(3):  # Retry 3 times
            response = None
            try:
                logger.debug(f"trial: {trial}")
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-2024-04-09",
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
                    implemented_udf = self.verify_syntax_correctness(implemented_udf, udf_vars, udf_name, py_func_signature, udf_description, n_obj, verify_syntax_correctness_base_prompt, self.program_with_pixels)
                    implemented_udf["udf_name"] = udf_name
                    implemented_udf["udf_signature"] = udf_signature
                    implemented_udf["udf_description"] = udf_description
                    implemented_udfs[idx] = implemented_udf
                    # implemented_udfs[idx] = copy.deepcopy(implemented_udf)
                    logger.info(f"[{idx}] semantic_interpretation: {implemented_udf['semantic_interpretation']}")
                    logger.info(f"[{idx}] function_implementation: {implemented_udf['function_implementation']}")
                    logger.info(f"[{idx}] kwargs: {implemented_udf.get('kwargs', {})}")
                    if self.save_generated_udf:
                        with open(os.path.join(self.save_udf_dir, f"{udf_name}_{idx}.json"), "w") as f:
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

    def verify_syntax_correctness(self, implemented_udf, udf_vars, udf_name, py_func_signature, udf_description, n_obj, verify_syntax_correctness_base_prompt, with_pixels, n_verify_samples=10):
        df_samples = self.construct_train_and_test_data(n_obj, n_verify_samples, with_images=self.program_with_pixels)
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
        for retry in range(5):
            try:
                if retry != 0:
                    response = self.client.chat.completions.create(
                        model="gpt-4-turbo-2024-04-09",
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
                exec(implemented_udf["function_implementation"], globals())
                udf_obj = globals()[py_func_name]
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
                    lambda row: udf_obj(row["img"], row["o1_oname"], row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"], row["o1_anames"], row["height"], row["width"], **kwargs), axis=1
                )
            else:
                result = df.apply(
                    lambda row: udf_obj(row["o1_oname"], row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"], row["o1_anames"], row["height"], row["width"], **kwargs), axis=1
                )
        elif n_obj == 2:
            if with_pixels:
                result = df.apply(
                    lambda row: udf_obj(row["img"], row["o1_oname"], row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"], row["o1_anames"], row["o2_oname"], row["o2_x1"], row["o2_y1"], row["o2_x2"], row["o2_y2"], row["o2_anames"], row["o1_o2_rnames"], row["o2_o1_rnames"], row["height"], row["width"], **kwargs), axis=1
                )
            else:
                result = df.apply(
                    lambda row: udf_obj(row["o1_oname"], row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"], row["o1_anames"], row["o2_oname"], row["o2_x1"], row["o2_y1"], row["o2_x2"], row["o2_y2"], row["o2_anames"], row["o1_o2_rnames"], row["o2_o1_rnames"], row["height"], row["width"], **kwargs), axis=1
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

    ##############################################################
    #############                                    #############
    ############# UDF Distillation (distilled-model) #############
    #############                                    #############
    ##############################################################
    def _distill_model(self, udf_signature, udf_description, gt_udf_name=None):
        """
        gt_udf_name: ground truth UDF name. If provided, compute llm's TP, FP, TN, FN, f1 score and test the trained model.
        """
        # Initialization for model distillation
        udf_name, udf_vars = parse_signature(udf_signature)
        self.udf_class = udf_name.lower()
        self.n_obj = len(udf_vars)
        assert self.n_obj in [1, 2], "n_obj must be 1 or 2"
        self.udf_description = udf_description

        # module_name, function_name = gt_udf_name.split(".")
        # module_name = "udfs.{}".format(module_name)
        # module = importlib.import_module(module_name)
        # self.gt_udf = getattr(module, function_name)
        self.gt_udf_name = gt_udf_name

        # NOTE: LLM doesn't generate labels in some cases, so we need to double the number of samples (i.e., self.n_train * 2) to ensure we have enough training samples
        if gt_udf_name:
            self.df_train, self.df_test = self.construct_train_and_test_data(self.n_obj, self.n_train_distill * 2, self.n_test_distill, with_images=True)
        else:
            self.df_train = self.construct_train_and_test_data(self.n_obj, self.n_train_distill * 2, with_images=True)

        self.llm_annotate_data()
        self.mlp_prepare_data()
        best_ckpt = self.train()
        if gt_udf_name:
            self.test()

        udf_dict = {}
        udf_dict["udf_name"] = udf_name
        udf_dict["udf_signature"] = udf_signature
        udf_dict["udf_description"] = udf_description
        udf_dict["semantic_interpretation"] = 'model'
        udf_dict["function_implementation"] = best_ckpt
        new_udf_candidate = UDFCandidate(id='model', payload=udf_dict)
        return [new_udf_candidate]

    def llm_annotate_data(self):
        labeled_data = defaultdict(list) # dictionary with 'train' and 'test' fields. Each field is a list of tuples (image_features, label)

        labeled_data_dir = os.path.join(self.config["model_dir"], "labeled_data", self.dataset, self.llm_method)
        labeled_data_path = os.path.join(labeled_data_dir, "udf-{}_run-{}_ntrain-{}_labeled_data.pt".format(self.udf_class, self.run_id, self.n_train_distill))
        os.makedirs(labeled_data_dir, exist_ok=True)
        if self.load_labeled_data and os.path.exists(labeled_data_path):
            logger.info("Loading labeled data from {}".format(labeled_data_path))
            labeled_data = torch.load(labeled_data_path)
            for data in labeled_data['train']:
                logger.debug("row: {}".format(data['row'].drop('img').to_dict()))
                logger.debug("base64_image: {}".format(data["base64_image"]))
                logger.debug("gt_label: {}, llm_label: {}".format(data["label"], data["llm_label"]))
            logger.debug("llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(labeled_data["metadata"]["llm_TP"], labeled_data["metadata"]["llm_FP"], labeled_data["metadata"]["llm_TN"], labeled_data["metadata"]["llm_FN"], labeled_data["metadata"]["llm_f1"]))
            if "test" in labeled_data:
                logger.debug("test_pos: {}, test_neg: {}".format(labeled_data["metadata"]["test_pos"], labeled_data["metadata"]["test_neg"]))
        else:
            self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN = 0, 0, 0, 0
            # Training and validation data
            self.label_count = 0
            for _, row in self.df_train.iterrows():
                try:
                    gt_label = self._get_gt_label(row)
                    logger.debug("gt_label: {}".format(gt_label))
                    # Read and crop frame
                    logger.debug("row: {}".format(row.drop('img').to_dict()))
                    frame, image_size = self.frame_processing_for_model(row)
                    if frame is None:
                        continue
                    llm_label, base64_image, image_prompt = self._llm_annotate_frame(frame, image_size, row, gt_label)
                    labeled_data['train'].append({"label": gt_label, "llm_label": llm_label, "base64_image": base64_image, "image_prompt": image_prompt, "row": row})
                    if self.label_count >= self.n_train_distill:
                        break
                except Exception as e:
                    logger.debug("Error: {}".format(e))
                    continue
            if self.gt_udf_name is not None:
                llm_f1 = 2*self.llm_TP/(2*self.llm_TP+self.llm_FP+self.llm_FN) if 2*self.llm_TP+self.llm_FP+self.llm_FN > 0 else 0.0
                logger.debug("llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN, llm_f1))
            else:
                llm_f1 = -1
            labeled_data["metadata"] = {"llm_TP": self.llm_TP, "llm_FP": self.llm_FP, "llm_TN": self.llm_TN, "llm_FN": self.llm_FN, "llm_f1": llm_f1}

            # Test data
            if self.gt_udf_name is not None:
                for _, row in self.df_test.iterrows():
                    try:
                        gt_label = self._get_gt_label(row)
                        labeled_data['test'].append({"label": gt_label, "row": row})
                    except Exception as e:
                        logger.debug("Error: {}".format(e))
                        continue
                pos_count = sum([1 for data in labeled_data['test'] if data["label"] == 1])
                neg_count = sum([1 for data in labeled_data['test'] if data["label"] == 0])
                logger.debug("test_pos: {}, test_neg: {}".format(pos_count, neg_count))
                labeled_data["metadata"]["test_pos"] = pos_count
                labeled_data["metadata"]["test_neg"] = neg_count
            # save labeled_data to a file
            if self.save_labeled_data:
                logger.info("Saving labeled data to {}".format(labeled_data_path))
                torch.save(labeled_data, labeled_data_path)
        self.labeled_data = labeled_data

    def mlp_prepare_data(self):
        splits = ['train', 'test'] if self.gt_udf_name is not None else ['train']
        for split in splits:
            logger.info("Processing {} data".format(split))
            idx_to_remove = []
            for i in range(len(self.labeled_data[split])):
                try:
                    data = self.labeled_data[split][i]
                    row = data['row']
                    logger.debug("row: {}".format(row.drop('img').to_dict()))
                    frame, image_size = self.frame_processing_for_model(row)
                    if frame is None: # failed to read the frame
                        idx_to_remove.append(i)
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_features = self.extract_features(frame, row, image_size)
                    self.labeled_data[split][i]["image_features"] = image_features
                except Exception as e:
                    logger.debug("Error: {}".format(e))
                    idx_to_remove.append(i)
                    continue
            for i in reversed(idx_to_remove):
                del self.labeled_data[split][i]

        # use 20% of the training data as validation data
        train_dataset = CustomImageDataset(self.labeled_data['train'], train=True)
        train_set_size = int(len(train_dataset) * 0.8)
        valid_set_size = len(train_dataset) - train_set_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(self.run_id))

        class_counts = [sum(data["llm_label"] == i for data in self.labeled_data['train']) for i in range(2)]
        self.class_weights = torch.tensor([1.0 / count for count in class_counts]).to(self.device)
        logger.debug("class_counts: {}, class_weights: {}".format(class_counts, self.class_weights))

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
        if self.gt_udf_name is not None:
            test_dataset = CustomImageDataset(self.labeled_data['test'], train=False)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    def train(self, active_learning_round=-1):
        # logger.debug("mlp_config: {}".format(mlp_config))
        mlp_dim_in = self.dim_in if self.n_obj == 1 else self.dim_in * 3
        logger.debug("mlp_dim_in: {}".format(mlp_dim_in)) # should be 512 for clip-vit-base-patch32
        self.checkpoint_root = os.path.join(self.config["model_dir"], "model_udf", self.dataset, f"{self.llm_method}_{self.mlp_method}", self.udf_class)
        if active_learning_round >= 0:
            self.checkpoint_root = os.path.join(self.checkpoint_root, f"active_learning_round_{active_learning_round}")
        self.checkpoint_filename = "udf-{}_run-{}_ntrain-{}".format(self.udf_class, self.run_id, self.n_train_distill)
        os.makedirs(self.checkpoint_root, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=self.checkpoint_filename,
            monitor="val_loss",
            mode="min",
        )
        callbacks=[checkpoint_callback]
        earlystopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
        callbacks.append(earlystopping_callback)

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
        best_ckpt = checkpoint_callback.best_model_path
        logger.debug("Best model checkpoint: {}".format(best_ckpt))
        # best_mlp_model = mlp.MLP.load_from_checkpoint(best_ckpt)
        return best_ckpt

    def test(self):
        # Inference:
        # model = mlp.MLP.load_from_checkpoint(self.dim_in, 2)
        logger.debug("test with last model: ")
        self.trainer.test(self.mlp_model, dataloaders=self.test_loader)
        logger.debug("test with best model: ")
        self.trainer.test(ckpt_path="best", dataloaders=self.test_loader)

    def _get_gt_label(self, row):
        if self.gt_udf_name is None:
            return None
        if self.n_obj == 1:
            return int(self.gt_udf_name in row["o1_gt_anames"])
        else:
            return int(self.gt_udf_name in row["o1_o2_gt_rnames"])

    def frame_processing_for_model(self, row):
        # vid = row['vid']
        # fid = row['fid']
        # cap = cv2.VideoCapture(
        #     os.path.join(
        #         self.config['data_dir'],
        #         self.dataset,
        #         f'video_{str(vid//1000*1000).zfill(5)}-{str((vid//1000+1)*1000).zfill(5)}',
        #         f"video_{str(vid).zfill(5)}.mp4"
        #     )
        # )
        # cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        # ret, frame = cap.read()
        # if not ret:
        #     logger.debug("Failed to read the frame")
        #     return None, None
        # cap.release()
        frame = row['img']
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_size = frame.shape[:2]
        if self.n_obj == 1:
            x1, y1, x2, y2 = self.expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
            frame = frame[y1:y2, x1:x2]
        else:
            o1_x1, o1_y1, o1_x2, o1_y2 = self.expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
            o2_x1, o2_y1, o2_x2, o2_y2 = self.expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
            frame = frame[min(o1_y1, o2_y1):max(o1_y2, o2_y2), min(o1_x1, o2_x1):max(o1_x2, o2_x2)]
        # resize the frame to 224x224
        frame = cv2.resize(frame, (224, 224))
        return frame, image_size

    def expand_box(self,x1,y1,x2,y2,img_size,factor=1.5):
        H, W = img_size
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]

    def _llm_annotate_frame(self, frame, image_size, row, gt_label):
        # TODO: don't resize the frame here, resize it in the model
        # Convert the frame to a base 64 encoded image
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        logger.debug("base64_image: {}".format(base64_image))
        image_prompt = self._create_image_prompt(row, image_size)
        logger.debug("Image prompt: {}".format(image_prompt))
        response = completion_with_backoff(
            model="gpt-4-turbo-2024-04-09",
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
        logger.debug("Result: {}".format(result))
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

    def _create_image_prompt(self, row, image_size):
        image_prompt = "{}? Answer with 'yes' or 'no'.".format(self.replace_objects(self.udf_description.rstrip(string.punctuation), row, image_size))
        return image_prompt

    def replace_objects(self, input_string, row, image_size):
        # Find all occurrences of "o" followed by integers
        objects = re.findall(r'o\d+', input_string)
        # Sort the objects based on the integer part of the identifier
        sorted_objects = sorted(objects, key=lambda x: int(x[1:]))

        h, w = image_size
        if len(sorted_objects) == 1:
            new_string = input_string.replace(sorted_objects[0], f"{row['o1_oname']} {sorted_objects[0]} at {round(row['o1_x1']/w, 3), round(row['o1_y1']/h, 3), round(row['o1_x2']/w, 3), round(row['o1_y2']/h, 3)}")
        elif len(sorted_objects) == 2:
            new_string = input_string.replace(sorted_objects[0], f"{row['o1_oname']} {sorted_objects[0]} at {round(row['o1_x1']/w, 3), round(row['o1_y1']/h, 3), round(row['o1_x2']/w, 3), round(row['o1_y2']/h, 3)}")
            new_string = new_string.replace(sorted_objects[1], f"{row['o2_oname']} {sorted_objects[1]} at {round(row['o2_x1']/w, 3), round(row['o2_y1']/h, 3), round(row['o2_x2']/w, 3), round(row['o2_y2']/h, 3)}")

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
            # _, buffer = cv2.imencode('.jpg', frame_subject)
            # base64_frame_subject = base64.b64encode(buffer).decode('utf-8')
            # logger.debug("base64_frame_subject: {}".format(base64_frame_subject))
            # _, buffer = cv2.imencode('.jpg', frame_target)
            # base64_frame_target = base64.b64encode(buffer).decode('utf-8')
            # logger.debug("base64_frame_target: {}".format(base64_frame_target))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_subject = cv2.cvtColor(frame_subject, cv2.COLOR_BGR2RGB)
            frame_target = cv2.cvtColor(frame_target, cv2.COLOR_BGR2RGB)
            inputs = self.clip_processor(images=[frame, frame_subject, frame_target], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            outputs = outputs.reshape(-1)
        return outputs

    def _compute_new_box_after_crop(self, row, image_size):
        o1_x1, o1_y1, o1_x2, o1_y2 = self.expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
        o2_x1, o2_y1, o2_x2, o2_y2 = self.expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
        x_offset = min(o1_x1, o2_x1)
        y_offset = min(o1_y1, o2_y1)
        h_ratio = 224.0 / (max(o1_y2, o2_y2) - y_offset)
        w_ratio = 224.0 / (max(o1_x2, o2_x2) - x_offset)
        return (row['o1_x1'] - x_offset) * w_ratio, (row['o1_y1'] - y_offset) * h_ratio, (row['o1_x2'] - x_offset) * w_ratio, (row['o1_y2'] - y_offset) * h_ratio, (row['o2_x1'] - x_offset) * w_ratio, (row['o2_y1'] - y_offset) * h_ratio, (row['o2_x2'] - x_offset) * w_ratio, (row['o2_y2'] - y_offset) * h_ratio

    ###########################
    ######               ######
    ###### UDF Selection ######
    ######               ######
    ###########################
    def select(self, gt_udf_name, udf_candidate_list):
        return self._select(gt_udf_name, udf_candidate_list, with_pixels=self.program_with_pixels)


    def _select(self, gt_udf_name, udf_candidate_list, with_pixels):
        if len(udf_candidate_list) == 1:
            selected_udf_candidate = udf_candidate_list[0]
        else:
            udf_signature = udf_candidate_list[0].udf_signature
            # Step 4: Select the best UDF
            udf_name, udf_vars = parse_signature(udf_signature)
            n_obj = len(udf_vars)

            # Construct training data and test data
            # TODO: determine if we always need to use with_images=True
            # TODO: Too slow.
            df_train, df_test = self.construct_train_and_test_data(n_obj, self.n_train, self.n_test, with_images=with_pixels)

            # Dynamically import the ground truth UDF
            # TODO: I think we can remove this since we already materialized the ground truth UDF in database
            # Example: gt_udf = gt_behind.gt_0
            # module_name, function_name = gt_udf_name.split(".")
            # module_name = "udfs.{}".format(module_name)
            # module = importlib.import_module(module_name)
            # gt_udf = getattr(module, function_name)

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
                    # y_true = df_train.loc[labeled_index].apply(
                    #     lambda row: gt_udf(row["o1"]), axis=1
                    # )
                    y_true = pd.Series([gt_udf_name in anames for anames in df_train.loc[labeled_index]['o1_gt_anames']])
                elif n_obj == 2:
                    # y_true = df_train.loc[labeled_index].apply(
                    #     lambda row: gt_udf(row["o1"], row["o2"]), axis=1
                    # )
                    y_true = pd.Series([gt_udf_name in rnames for rnames in df_train.loc[labeled_index]['o1_o2_gt_rnames']])
                # log number of positive and negative samples
                logger.info(
                    "# positive: {}, # negative: {}".format(
                        sum(y_true), len(y_true) - sum(y_true)
                    )
                )
                # Update scores
                for i in range(len(udf_candidate_list)):
                    score, loss_t = self.compute_udf_score(
                        gt_udf_name,
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
                    gt_udf_name,
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

        if selected_udf_candidate.id != "model":
            # Transforms the function by removing **kwargs from the function signature and replacing the kwargs with the actual values
            selected_udf_candidate.function_implementation = transform_function(
                original_code=selected_udf_candidate.function_implementation,
                instantiation_dict=selected_udf_candidate.kwargs,
            )
        if self.save_generated_udf:
            with open(os.path.join(self.save_udf_dir, f"{udf_name}_selected.json"), "w") as f:
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

        df_sampled = df_train.loc[sampled_index]

        indices_to_remove = []
        for i, udf_candidate in enumerate(udf_candidate_list):
            if udf_candidate.id == "model":
                # distilled-model UDF
                best_ckpt = udf_candidate.function_implementation
                predictions = self.predict_with_data(df_sampled, best_ckpt, n_obj)
                logger.debug("predictions: {}".format(predictions))
                prediction_matrix.append(predictions)
            else:
                try:
                    # program-based UDF
                    # For each sampled row in df_train, construct o1 and o2
                    kwargs = {}
                    for k, v in udf_candidate.kwargs.items():
                        kwargs[k] = float(v)
                    py_func_name = "py_{}".format(udf_name)
                    exec(udf_candidate.function_implementation, globals())
                    udf_obj = globals()[py_func_name]
                    # TODO: may need to timeout if running for too long
                    result = self.exec_udf_with_data(df_sampled, udf_obj, kwargs, n_obj, with_pixels)
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

    def predict_with_data(self, df, ckpt, n_obj):
        df = df.reset_index(drop=True)
        # Predict the labels of all the data points
        checkpoint = torch.load(ckpt)
        hyper_parameters = checkpoint["hyper_parameters"]
        mlp_model = mlp.MLPProd(**hyper_parameters)
        mlp_model.load_state_dict(checkpoint["state_dict"])
        mlp_model.eval()
        mlp_model.to(self.device)

        if n_obj == 1:
            df_with_features = self.conn.execute(f"""
                SELECT any_value(d.feature) as feature
                FROM df,
                    '{self.attribute_features_dir}/*.parquet' d
                WHERE df.vid=d.vid AND df.fid=d.fid
                    AND df.o1_oid=d.o1_oid
                GROUP BY d.vid, d.fid, d.o1_oid
                ORDER BY df.index
            """).df()
        else:
            df_with_features = self.conn.execute(f"""
                SELECT any_value(d.feature) as feature
                FROM df,
                    '{self.relationship_features_dir}/*.parquet' d
                WHERE df.vid=d.vid AND df.fid=d.fid
                    AND df.o1_oid=d.o1_oid AND df.o2_oid=d.o2_oid
                GROUP BY d.vid, d.fid, d.o1_oid, d.o2_oid
                ORDER BY df.index
            """).df()
        predictions = []
        with torch.no_grad():
            for _, row in tqdm(df_with_features.iterrows()):
                feature = torch.tensor(row["feature"], dtype=torch.float32).to(self.device)
                pred = mlp_model(feature)
                predictions.append(bool(pred.item()))
        return predictions

    def compute_udf_score(
        self,
        gt_udf_name,
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
        if udf_candidate.id == "model":
                best_ckpt = udf_candidate.function_implementation
                y_pred = self.predict_with_data(df, best_ckpt, n_obj)
                if df_newly_labeled is not None:
                    y_pred_new = self.predict_with_data(df_newly_labeled, best_ckpt, n_obj)
        else:
            try:
                # For each sampled row in df, construct o1 and o2
                kwargs = {}
                for k, v in udf_candidate.kwargs.items():
                    kwargs[k] = float(v)
                py_func_name = "py_{}".format(udf_name)
                exec(udf_candidate.function_implementation, globals())
                udf_obj = globals()[py_func_name]
                y_pred = self.exec_udf_with_data(df, udf_obj, kwargs, n_obj, with_pixels)
                if df_newly_labeled is not None:
                    y_pred_new = self.exec_udf_with_data(df_newly_labeled, udf_obj, kwargs, n_obj, with_pixels).reset_index(drop=True)
            except Exception as e:
                logger.debug("ERROR: failed to execute udf_candidate {}: {}".format(udf_candidate.id, e))
                # y_pred = df.apply(lambda row: False, axis=1)
                y_pred = pd.Series([False] * len(df))
                if df_newly_labeled is not None:
                    y_pred_new = pd.Series([False] * len(df_newly_labeled))

        # Compute y_true and f1 score
        if n_obj == 1:
            # y_true = df.apply(lambda row: gt_udf_name in row["o1_gt_anames"], axis=1)
            y_true = pd.Series([gt_udf_name in o1_gt_anames for o1_gt_anames in df['o1_gt_anames']])
        elif n_obj == 2:
            # y_true = df.apply(lambda row: gt_udf_name in row["o1_o2_gt_rnames"], axis=1)
            y_true = pd.Series([gt_udf_name in o1_o2_gt_rnames for o1_o2_gt_rnames in df['o1_o2_gt_rnames']])
        # logger.debug(f"y_true: {y_true}, y_pred: {y_pred}")
        score = f1_score(y_true, y_pred, zero_division=1.0)
        logger.info(
            "positive: {}, negative: {}".format(sum(y_true), len(y_true) - sum(y_true))
        )

        # Compute y_true_new and num_misclassified
        if df_newly_labeled is not None:
            if n_obj == 1:
                # y_true_new = df_newly_labeled.apply(
                #     lambda row: gt_udf(row["o1_oname"], row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"], row["o1_anames"], row["height"], row["width"]), axis=1
                # )
                y_true_new = pd.Series([gt_udf_name in o1_gt_anames for o1_gt_anames in df_newly_labeled['o1_gt_anames']])
            elif n_obj == 2:
                # y_true_new = df_newly_labeled.apply(
                #     lambda row: gt_udf(row["o1_oname"], row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"], row["o1_anames"], row["o2_oname"], row["o2_x1"], row["o2_y1"], row["o2_x2"], row["o2_y2"], row["o2_anames"], row["o1_o2_rnames"], row["o2_o1_rnames"], row["height"], row["width"]), axis=1
                # )
                y_true_new = pd.Series([gt_udf_name in o1_o2_gt_rnames for o1_o2_gt_rnames in df_newly_labeled['o1_o2_gt_rnames']])
            # Count the number of misclassifications for the new samples
            # logger.debug(f"y_true_new: {y_true_new}, y_pred_new: {y_pred_new}")
            num_misclassified = np.sum(np.array(y_true_new != y_pred_new) * 1)
            return score, num_misclassified
        else:
            return score

    def construct_train_and_test_data(self, n_obj, n_train=None, n_test=None, with_images=False):
        if with_images:
            return self._construct_train_and_test_data_with_images(n_obj, n_train, n_test)
        else:
            return self._construct_train_and_test_data_without_images(n_obj, n_train, n_test)

    def _construct_train_and_test_data_without_images(self, n_obj, n_train=None, n_test=None):
        # Construct training data and test data
        self.conn.execute(f"SELECT setseed({self.run_id / 100})")
        if n_train is None and n_test is None:
            if n_obj == 1:
                sql = "SELECT * FROM one_object"
                df_filtered = self.conn.execute(sql).df()
            elif n_obj == 2:
                sql = "SELECT * FROM two_objects"
                df_filtered = self.conn.execute(sql).df()
            else:
                raise ValueError("Number of objects not supported: {}".format(n_obj))
        else:
            if n_obj == 1:
                sql = f"""
                    SELECT *
                    FROM one_object
                    ORDER BY random()
                    LIMIT {n_train + n_test if n_test else n_train}
                """
                # logger.debug(sql)
                df_filtered = self.conn.execute(sql).df()
                # df_filtered["o1"] = df_filtered.apply(lambda row: row.to_dict(), axis=1)
            elif n_obj == 2:
                sql = f"""
                    SELECT *
                    FROM two_objects
                    ORDER BY random()
                    LIMIT {n_train + n_test if n_test else n_train}
                """
                # logger.debug(sql)
                df_filtered = self.conn.execute(sql).df()
                # df_filtered["o1"] = df_filtered.apply(
                #     lambda row: {
                #         col.split("_", 1)[1]: row[col]
                #         for col in df_filtered.columns
                #         if col.startswith("o1_")
                #     },
                #     axis=1,
                # )
                # df_filtered["o2"] = df_filtered.apply(
                #     lambda row: {
                #         col.split("_", 1)[1]: row[col]
                #         for col in df_filtered.columns
                #         if col.startswith("o2_")
                #     },
                #     axis=1,
                # )
            else:
                raise ValueError("Number of objects not supported: {}".format(n_obj))
        if n_test:
            df_train = df_filtered[:n_train]
            df_train = df_train.reset_index(drop=True)
            df_test = df_filtered[n_train:]
            df_test = df_test.reset_index(drop=True)
            return df_train, df_test
        else:
            df_filtered = df_filtered.reset_index(drop=True)
            return df_filtered

    def _construct_train_and_test_data_with_images(self, n_obj, n_train=None, n_test=None):
        # Construct training data and test data
        # dataframe consists of columns: img, o1 [, o2]
        if n_test:
            df_train, df_test = self._construct_train_and_test_data_without_images(n_obj, n_train, n_test)
            # construct the img column
            for df in [df_train, df_test]:
                df["img"] = df.progress_apply(self.frame_processing_for_program, axis=1)
            return df_train, df_test
        else:
            df = self._construct_train_and_test_data_without_images(n_obj, n_train)
            df["img"] = df.progress_apply(self.frame_processing_for_program, axis=1)
            return df

    def frame_processing_for_program(self, row):
        vid = row.vid
        cap = cv2.VideoCapture(
            os.path.join(
                self.config['data_dir'],
                self.dataset,
                f'video_{str(vid//1000*1000).zfill(5)}-{str((vid//1000+1)*1000).zfill(5)}',
                f"video_{str(vid).zfill(5)}.mp4"
            )
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, row.fid)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            raise ValueError("Failed to read the frame")
        cap.release()
        return frame

    def _append_features(self, df, n_obj):
        if n_obj == 1:
            df = self.conn.execute(f"""
                SELECT
                    df.vid AS vid, df.fid AS fid, df.o1_oid AS o1_oid, df.o1_oname AS o1_oname, df.o1_x1 AS o1_x1, df.o1_y1 AS o1_y1, df.o1_x2 AS o1_x2, df.o1_y2 AS o1_y2, df.o1_gt_anames AS o1_gt_anames, df.o1_anames AS o1_anames, df.height AS height, df.width AS width,
                    f.feature AS feature
                FROM df
                JOIN '{self.attribute_features_dir}/*.parquet' f
                ON f.vid = df.vid AND f.fid = df.fid AND f.o1_oid = df.o1_oid AND f.o1_x1 = df.o1_x1 AND f.o1_y1 = df.o1_y1 AND f.o1_x2 = df.o1_x2 AND f.o1_y2 = df.o1_y2
            """).df()
        else:
            df = self.conn.execute(f"""
                SELECT
                    df.vid AS vid, df.fid AS fid,
                    df.o1_oid AS o1_oid, df.o1_oname AS o1_oname, df.o1_x1 AS o1_x1, df.o1_y1 AS o1_y1, df.o1_x2 AS o1_x2, df.o1_y2 AS o1_y2, df.o1_anames AS o1_anames,
                    df.o2_oid AS o2_oid, df.o2_oname AS o2_oname, df.o2_x1 AS o2_x1, df.o2_y1 AS o2_y1, df.o2_x2 AS o2_x2, df.o2_y2 AS o2_y2, df.o2_anames AS o2_anames,
                    df.o1_o2_rnames AS o1_o2_rnames, df.o2_o1_rnames AS o2_o1_rnames, df.o1_o2_gt_rnames AS o1_o2_gt_rnames,
                    df.height AS height, df.width AS width,
                    f.feature AS feature
                FROM df
                JOIN '{self.relationship_features_dir}/*.parquet' f
                ON f.vid = df.vid AND f.fid = df.fid AND f.o1_oid = df.o1_oid AND f.o1_x1 = df.o1_x1 AND f.o1_y1 = df.o1_y1 AND f.o1_x2 = df.o1_x2 AND f.o1_y2 = df.o1_y2 AND f.o2_oid = df.o2_oid AND f.o2_x1 = df.o2_x1 AND f.o2_y1 = df.o2_y1 AND f.o2_x2 = df.o2_x2 AND f.o2_y2 = df.o2_y2
            """).df()
        return df

    def compute_best_test_score(self, gt_udf_name, udf_candidate_list):
        return self._compute_best_test_score(gt_udf_name, udf_candidate_list, with_pixels=self.program_with_pixels)

    def _compute_best_test_score(self, gt_udf_name, udf_candidate_list, with_pixels):
        udf_signature = udf_candidate_list[0].udf_signature
        # Step 4: Select the best UDF
        udf_name, udf_vars = parse_signature(udf_signature)
        n_obj = len(udf_vars)

        # Construct training data and test data
        _, df_test = self.construct_train_and_test_data(n_obj, self.n_train, self.n_test, with_images=True)

        # Dynamically import the ground truth UDF
        # Example: gt_udf = gt_behind.gt_0
        # module_name, function_name = gt_udf_name.split(".")
        # module_name = "udfs.{}".format(module_name)
        # module = importlib.import_module(module_name)
        # gt_udf = getattr(module, function_name)

        # compute test F1 score
        logger.info("compute test F1 score")
        best_test_score = -1
        for i in range(len(udf_candidate_list)):
            test_score = self.compute_udf_score(
                gt_udf_name,
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

# class CodeUDFWithPixelsProposer(UDFProposer):
#     llm_method = "gpt4v"
#     mlp_method = "three_clip"
#     program_with_pixels = True
#     def __init__(
#         self,
#         config,
#         prompt_config,
#         config_list,
#         registered_functions,
#         dataset,
#         labeling_budget,
#         num_interpretations,
#         num_parameter_search,
#         program_with_pixels,
#         query_id,
#         run_id,
#         num_workers,
#         save_generated_udf,
#         save_labeled_data,
#         load_labeled_data,
#         n_train_distill,
#         selection_strategy,
#         allow_kwargs_in_udf=False,
#     ):
#         super().__init__(
#             config,
#             prompt_config,
#             config_list,
#             registered_functions,
#             dataset,
#             labeling_budget,
#             num_interpretations,
#             num_parameter_search,
#             query_id,
#             run_id,
#             num_workers,
#             save_generated_udf,
#             save_labeled_data,
#             load_labeled_data,
#             n_train_distill,
#             selection_strategy,
#             allow_kwargs_in_udf,
#         )
#         self.n_train = 1000
#         self.n_test = 1000

#         self.save_udf_dir_lambda = lambda udf_name: os.path.join(
#             self.config["output_dir"],
#             "compare_code_and_model_udf",
#             self.dataset,
#             (
#                 "budget-{}_ninterp-{}_nparams-{}_with_kwargs".format(
#                     self.labeling_budget, self.num_interpretations, self.num_parameter_search
#                 )
#                 if self.allow_kwargs_in_udf
#                 else "budget-{}_ninterp-{}_without_kwargs".format(
#                     self.labeling_budget, self.num_interpretations
#                 )
#             ),
#             f"run-{self.run_id}",
#             udf_name,
#         )

#     def select(self, gt_udf_name, udf_candidate_list):
#         return self._select(gt_udf_name, udf_candidate_list, with_pixels=True)

#     def compute_best_test_score(self, gt_udf_name, udf_candidate_list):
#         return self._compute_best_test_score(gt_udf_name, udf_candidate_list, with_pixels=True)
