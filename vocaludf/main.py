from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import autogen
from src.utils import program_to_dsl, dsl_to_program
import yaml
import json
from parser import parse
import pyparsing as pp
import os
from vocaludf.utils import duckdb_execute_cache_sequence, duckdb_execute_clevrer_cache_sequence
import time
import duckdb
import logging
import argparse
from duckdb_dir.udf import register_udf
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import re
import io, tokenize
from collections import defaultdict

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key +"}}", value.replace('"', "'"))
    return text

def parse_signature(signature):
    tokens = list(tokenize.generate_tokens(io.StringIO(signature).readline))
    udf_name = tokens[0].string
    udf_vars = [token for token in tokens[2:-3] if token.string not in [',','=']]
    return udf_name, udf_vars

class QueryParser:
    def __init__(self, registered_functions):
        system_message = replace_slot(
            " ".join([
                prompt_config["system_prompt"]["parse_query_overall"],
                prompt_config["system_prompt"]["dsl_definition"],
                prompt_config["system_prompt"]["udf_definition"],
                prompt_config["system_prompt"]["registered_udfs"],
                prompt_config["system_prompt"]["parse_query_details"],
            ]),
            {"functions": "\n".join(["{}: {}".format(func["signature"], func["description"]) for func in registered_functions])}
        )

        self.parser = autogen.AssistantAgent(
            name="parser",
            system_message=system_message,
            llm_config={
                "config_list": config_list,
                "timeout": 120,
                "temperature": 0,
            }
        )

        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and ("yes" in x.get("content", "").rstrip().lower() or "no" in x.get("content", "").rstrip().lower()),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False
            }
        )

        @self.user_proxy.register_for_execution()
        @self.parser.register_for_llm(description="Verify syntax correctness of input query.")
        def verify_syntax(
            query: Annotated[str, "Input query written in DSL to be verified."],
        ) -> str:
            global parsed_program
            try:
                parsed_program = parse().parseString(query, parseAll=True).as_dict()
                # results_dump = parse().parseString(s, parseAll=True).dump()
            except (pp.ParseException, pp.ParseSyntaxException) as err:
                return err.explain()
            except Exception as e:
                error_message = query + "failed:\n" + str(e)
                return error_message
            else:
                return parsed_program

    def parse(self, user_query):
        chat_result = self.user_proxy.initiate_chat(
            self.parser,
            # message="Two objects move from far to close, then to far again",
            # message="In lane 1, o1 accelerates rapidly and then o2 accelerates rapdily",
            message=user_query,
        )
        chat_messages = self.user_proxy.chat_messages[self.parser]
        print("chat_messages", chat_messages)
        flag = chat_messages[-1]['content'].strip().lower()
        print(flag)
        return flag

class UDFProposer:
    # Propose new UDFs and generate semantic interpretations
    def __init__(self, config, prompt_config, registered_functions, dataset):
        self.config = config
        self.prompt_config = prompt_config
        self.registered_functions = registered_functions
        self.dataset = dataset
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.get_schema()

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
        obj_cols = [("class_name", "str"), ("x1", "float"), ("y1", "float"), ("x2", "float"), ("y2", "float")] + list((key, "str") for key in self.attribute_schema.keys())
        obj_cols_str = ", ".join([f"{col[0]}: {col[1]}" for col in obj_cols])
        # TODO: Add relationship to the prompt
        self.schema_prompt = f"Each object is represented by a dictionary of {{ {obj_cols_str} }}. "
        possible_values = ", ".join(["'" + c + "'" for c in self.object_schema])
        self.schema_prompt += f"'class_name' can be one of the following: [{possible_values}]. "
        image_h = self.config[self.dataset]["height"]
        image_w = self.config[self.dataset]["width"]
        self.schema_prompt += f"x1, y1, x2, y2 are the top-left and bottom-right coordinates of the bounding box. The coordinates are relative to the top-left corner of a {image_w}x{image_h} frame. "
        for key, values in self.attribute_schema.items():
            possible_values = ", ".join(["'" + v + "'" for v in values])
            self.schema_prompt += f"'{key}' can be one of the following: [{possible_values}]. "
        logger.debug("Schema prompt: {}".format(self.schema_prompt))

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def propose(self, user_query):
        # Step 1: propose new UDFs
        logger.info("Proposing new UDFs")
        system_message = replace_slot(
            " ".join([
                self.prompt_config["system_prompt"]["dsl_definition"],
                self.prompt_config["system_prompt"]["udf_definition"],
                self.prompt_config["system_prompt"]["registered_udfs"],
            ]),
            {"functions": "\n".join(["{}: {}".format(func["signature"], func["description"]) for func in registered_functions])}
        )
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": " ".join([
                    f"User query: {user_query}",
                    self.prompt_config["prompt"]["propose_udfs"],
                ])
            },
        ]
        logger.debug(f"messages: {messages}", )
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=self.config["udf_proposer"]["temperature"],
            top_p=self.config["udf_proposer"]["top_p"],
            max_tokens=512,
            seed=self.propose.retry.statistics['attempt_number']
        )
        proposed_functions = json.loads("\n\n".join(re.findall(r"```json\n(.*?)```", response.choices[0].message.content, re.DOTALL)))
        logger.info("Proposed functions: {}".format(proposed_functions))
        # Step 2: verify functions (i.e., whether they can be constructed out of existing ones)
        # TODO: Implement this
        return proposed_functions

    def implement(self, udf_signature, udf_description):
        # Step 3: generate semantic interpretations
        # TODO: Consider kwargs
        num_interpretations = self.config["interpretation_generator"]["num_interpretations"]
        generate_udfs_base_prompt = self.prompt_config["prompt"]["generate_udfs"]
        udf_name, udf_vars = parse_signature(udf_signature)
        logger.info(f"Implementing UDF: {udf_signature}, with {num_interpretations} semantic interpretations")
        generate_udfs_prompt = replace_slot(generate_udfs_base_prompt, {
            "num_interpretations": num_interpretations,
            "udf_signature": udf_signature,
            "udf_description": udf_description,
            "schema_info": self.schema_prompt,
            "n_obj": "one object" if len(udf_vars) == 1 else "two objects",
        })
        logger.debug("generate_udfs_prompt: {}".format(generate_udfs_prompt))
        for _ in range(3): # Retry 3 times
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert programming assistant."
                        },
                        {
                            "role": "user",
                            "content": generate_udfs_prompt
                        }
                    ],
                    temperature=self.config["udf_generator"]["temperature"],
                    top_p=self.config["udf_generator"]["top_p"],
                )
                implemented_udfs = json.loads("\n\n".join(re.findall(r"```json\n(.*?)```", response.choices[0].message.content, re.DOTALL)))["answer"]
                os.makedirs(os.path.join(self.config["output_dir"], "udf_generation", udf_name), exist_ok=True)
                for idx, implemented_udf in enumerate(implemented_udfs):
                    semantic_interpretation = implemented_udf["semantic_interpretation"]
                    function_implementation = implemented_udf["function_implementation"]
                    with open(os.path.join(self.config["output_dir"], "udf_generation", udf_name, "{}_{}.json".format(udf_name, idx)), "w") as f:
                        json.dump({
                            "udf_name": udf_name,
                            "udf_signature": udf_signature,
                            "udf_description": udf_description,
                            "semantic_interpretation": semantic_interpretation,
                            "function_implementation": function_implementation
                        }, f)
                    logger.info(f"[{idx}] semantic_interpretation: {semantic_interpretation}")
                    logger.info(f"[{idx}] function_implementation: {function_implementation}")
                return implemented_udfs
            except Exception as e:
                print("ERROR: failed to implement UDF", e)
                print(response)

class QueryExecutor:
    def __init__(self, config, inputs_table_name, registered_functions):
        self.config = config
        self.inputs_table_name = inputs_table_name
        self.conn = duckdb.connect(database=os.path.join(self.config['db_dir'], 'annotations.duckdb'), read_only=True)
        self.registered_functions = registered_functions
        for func in self.registered_functions:
            signature = func['signature']
            udf_name, udf_vars = parse_signature(signature)
            python_func_name = func['python_function'].split()[1].split('(')[0]
            python_func_args = func['python_function'].split('(')[1].split(')')[0].split(',')
            # TODO: Add type annotation to the function
            python_arg_str = ", ".join(f"{arg}: dict" for arg in python_func_args)
            python_header_type_annotated = f"def {python_func_name}({python_arg_str}) -> bool:"
            # Remove the first line of the function definition
            python_function = python_header_type_annotated + '\n' + '\n'.join(func['python_function'].split('\n')[1:])
            exec(python_function)
            exec(f"self.conn.create_function('{udf_name}', {python_func_name})")
            logger.debug(f"Registered function: {func['signature']}")

    def run(self, program):
        if self.inputs_table_name == "Obj_clevrer":
            # input_vids = 10000
            input_vids = 1000
        else:
            raise ValueError("Unknown inputs_table_name: {}".format(self.inputs_table_name))
        logger.info("Running query: {}".format(program['query']))
        _start = time.time()
        memo = [{} for _ in range(72159)] # Not used
        # TODO: Update duckdb_execute_clevrer_cache_sequence so that UDFs take tables (i.e., dict) as input
        result, new_memo = duckdb_execute_clevrer_cache_sequence(self.conn, program['query'], memo, self.inputs_table_name, input_vids, table_as_input_to_udf=True)
        logger.info("Time to execute query: {}".format(time.time() - _start))
        result = sorted(result)
        logger.info("output vids: {}".format(result))
        labels = []
        for i in range(input_vids):
            if i in result:
                labels.append(1)
            else:
                labels.append(0)
        logger.info("predictions: {}".format(labels))

if __name__ == "__main__":
    config = yaml.safe_load(open("/home/enhao/VOCAL-UDF/configs/config.yaml", "r"))

    parser = argparse.ArgumentParser()
    # parser.add_argument("--udf_class", type=str, help="UDF class name we want to generate")
    # parser.add_argument("--n_obj", type=int, help="number of objects in the UDF signature")
    # parser.add_argument("--udf_description", type=str, help="UDF description")
    # parser.add_argument("--gt_udf_impl", type=str, help="ground truth UDF implementation")
    # parser.add_argument("--udf_generation_name", type=str, help="name of the function that GPT will generate")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--log_name_suffix", type=str, help="log name suffix")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    # parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    # parser.add_argument("--stage", type=str, help="stage of the experiment (udf_generation, udf_selection, or all)")

    args = parser.parse_args()
    # gt_udf_impl = args.gt_udf_impl
    # udf_class = args.udf_class
    # n_obj= args.n_obj
    # udf_description = args.udf_description
    # udf_generation_name = "py_{}".format(udf_class)
    dataset = args.dataset
    log_name_suffix = args.log_name_suffix
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    # num_interpretations = args.num_interpretations
    # stage = args.stage

    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    # os.makedirs(config['log_dir'], exist_ok=True)
    # # os.makedirs(os.path.join('/home/enhao/EQUI-VOCAL/outputs/udf_generation', udf_class), exist_ok=True)

    # # Create a file handler that logs even debug messages
    # file_handler = logging.FileHandler(os.path.join(config['log_dir'], '{}_{}.log'.format(gt_udf_impl, log_name_suffix)), mode='w')
    # file_handler.setLevel(logging.DEBUG)

    # # Create a console handler with a higher log level
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)

    # # Create formatters and add them to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # # Add the handlers to the logger
    # logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    # Load LLM inference endpoints from an env variable or a file
    # See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
    # and OAI_CONFIG_LIST_sample
    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        },
    )

    prompt_config = yaml.load(open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"), Loader=yaml.FullLoader)

    registered_functions = json.load(open("registered_udfs.json", "r"))['test']

    user_query = "For at least 15 frames, an object o2 is to the left of a red object o0, and another object o1 is in the front of o2. Then, for a minimum of 5 frames, o1 is far from o2 and o0 is to the left of o2. Finally, o2 is behind o0, and o1 is made of metal."
    # user_query = "Two objects move from far to close, then to far again"

    up = UDFProposer(config, prompt_config, registered_functions, dataset)
    qp = QueryParser(registered_functions)
    flag = qp.parse(user_query)
    # Parse query
    while "no" in flag:
        # Step 1: propose new UDFs
        proposed_functions = up.propose(user_query)
        for udf_signature, udf_description in proposed_functions.items():
            # # Step 2: generate semantic interpretations
            # interpretations = up.generate_semantic_interpretation(udf_signature, udf_description)
            # # Step 3: generate UDFs
            # for iid, interpretation in enumerate(interpretations):
            #     function_implementation = up.implement(interpretation, udf_signature, udf_description, iid)
            implemented_udfs = up.implement(udf_signature, udf_description)
            # Step 4: Select the best UDF
            # TODO: Implement this
            # Assume now that the best UDF is the first one
            best_impl = implemented_udfs[0]
            logger.info("Best {} implementation: {}".format(udf_signature, best_impl["function_implementation"]))
            # Step 5: Register the UDF
            new_udf = {
                "signature": udf_signature,
                "description": udf_description,
                "semantic_interpretation": best_impl["semantic_interpretation"], # New field. Unsure if we need this
                "python_function": best_impl["function_implementation"]
            }
            registered_functions.append(new_udf)
        # Step 6: Re-parse the query
        qp = QueryParser(registered_functions)
        flag = qp.parse(user_query)
    if "yes" in flag:
        qe = QueryExecutor(config, "Obj_clevrer", registered_functions)
        qe.run(parsed_program)
    else:
        raise ValueError("Invalid response from user_proxy")
