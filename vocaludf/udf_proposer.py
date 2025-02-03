from typing_extensions import Annotated
import autogen
from autogen import gather_usage_summary
from typing import List
import os
import logging
from collections import defaultdict
from vocaludf.utils import (
    replace_slot,
    parse_signature,
    MODEL_COST,
    SharedResources,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class UDFProposer:
    def __init__(self, shared_resources: SharedResources):
        # Shared resources
        self.shared_resources = shared_resources
        self.config = shared_resources.config
        self.prompt_config = shared_resources.prompt_config
        self.dataset = shared_resources.dataset
        self.registered_functions = shared_resources.registered_functions
        self.object_domain = shared_resources.object_domain
        self.openai_model_name = shared_resources.openai_model_name
        self.run_id = shared_resources.run_id

        self.cost_estimation = defaultdict(float)

    def get_cost_estimation(self):
        return self.cost_estimation

    def propose(self, user_query):
        self.proposed_functions = {}
        # Step 1: propose new UDFs
        logger.info("Proposing new UDFs")
        if self.dataset in ["clevrer", "charades", "cityflow"]:  # video dataset
            dsl_definition_prompt = self.prompt_config["dsl_definition"]
        elif self.dataset in ["clevr", "gqa", "vaw"]:  # image dataset
            dsl_definition_prompt = self.prompt_config["dsl_definition_image"]
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")
        system_message = replace_slot(
            " ".join(
                [
                    dsl_definition_prompt,
                    self.prompt_config["udf_definition"]["without_object"] if self.dataset in ["clevr", "clevrer", "vaw", "cityflow"] else self.prompt_config["udf_definition"]["with_object"],
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
                "config_list": [{
                    'model': self.openai_model_name,
                    'api_key': os.getenv("OPENAI_API_KEY"),
                }],
                "timeout": 120,
                "temperature": self.config["udf_proposer"]["temperature"],
                "seed": self.run_id,
                "top_p": self.config["udf_proposer"]["top_p"],
                "max_tokens": 512,
                "cache_seed": None,
            },
        )

        # TODO: Add cost estimation
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "")
            and "terminate" in x.get("content", "").rstrip().lower(),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            # code_execution_config={"work_dir": "coding", "use_docker": False},
        )

        @user_proxy.register_for_execution()
        @udf_proposer.register_for_llm(
            description="Verify syntax correctness of proposed UDFs."
        )
        def verify_syntax(
            proposed_functions: Annotated[
                List[List[str]],
                "A list of proposed functions where proposed_functions[i] = [signature_i, description_i]. 'signature_i' represents the function signature 'function(args)', and 'description_i' contains the function description that starts with the word 'whether' and captures any specific definition as mentioned in the user query.",
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
                    for func in proposed_functions:
                        self.proposed_functions[func[0]] = func[1]
                    return "Success"
            except Exception as e:
                return "Error: " + str(e)

        logger.debug(f"system_message: {system_message}")

        user_proxy.initiate_chat(
            udf_proposer,
            message=f"User query: {user_query}",
        )
        usage_summary = gather_usage_summary([udf_proposer, user_proxy])["usage_including_cached_inference"][self.openai_model_name]
        self.cost_estimation["propose_udfs"] += usage_summary["prompt_tokens"] * MODEL_COST[self.openai_model_name][0] + usage_summary["completion_tokens"] * MODEL_COST[self.openai_model_name][1]

        try:
            logger.info(
                "Proposed functions: {}".format(self.proposed_functions)
            )  # key: signature, value: description
            registered_function_names = set(
                [
                    registered_function["signature"].split("(")[0].lower()
                    for registered_function in self.registered_functions
                ]
            )
            logger.info("filtering out functions that are already registered")
            for key in list(self.proposed_functions.keys()):
                if key.split("(")[0].lower() in registered_function_names:
                    logger.info(f"filtering out {key}")
                    del self.proposed_functions[key]
            # Step 2: verify functions (i.e., whether they can be constructed out of existing ones)
            # TODO: Implement this
            return self.proposed_functions
        except Exception as e:
            f"Error: {e}"
            return {}
