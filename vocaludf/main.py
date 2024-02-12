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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        new_text = text.replace("{{" + key +"}}", value.replace('"', "'"))
    return new_text

class QueryParser:
    def __init__(self):
        system_message = replace_slot(
            prompt_config["system_prompt"]["parse_query"],
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
    def __init__(self, prompt_config, registered_functions):
        self.prompt_config = prompt_config
        self.registered_functions = registered_functions
    def propose_udfs(self, user_query):
        pass

class QueryExecutor:
    def __init__(self, inputs_table_name, registered_functions):
        self.inputs_table_name = inputs_table_name
        self.conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=True)
        self.registered_functions = registered_functions
        for func in self.registered_functions:
            exec(func['python_function'])
            python_func_name = func['python_function'].split()[1].split('(')[0]
            exec(f"self.conn.create_function(func['signature'].split('(')[0], {python_func_name})")
            logger.debug(f"Registered function: {func['signature']}")
    def run(self, program):
        if self.inputs_table_name == "Obj_clevrer":
            input_vids = 10000
        else:
            raise ValueError("Unknown inputs_table_name: {}".format(self.inputs_table_name))
        _start = time.time()
        memo = [{} for _ in range(72159)] # Not used
        result, new_memo = duckdb_execute_clevrer_cache_sequence(self.conn, program['query'], memo, self.inputs_table_name, input_vids)
        print("Time to execute query: {}".format(time.time() - _start))
        result = sorted(result)
        labels = []
        for i in range(input_vids):
            if i in result:
                labels.append(1)
            else:
                labels.append(0)

class UDFGenerator:
    def __init__(self, prompt_config, registered_functions):
        self.prompt_config = prompt_config
        self.registered_functions = registered_functions

    def propose_udfs(self, parsed_program):
        pass

    def generate_semantic_interpretation(generate_semantic_interpretation_prompt):
        logger.info("generate_semantic_interpretation_prompt: {}".format(generate_semantic_interpretation_prompt))
        try:
            response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                "role": "system",
                "content": "You are an expert programming assistant."
                },
                {
                "role": "user",
                "content": generate_semantic_interpretation_prompt
                }
            ],
            temperature=0.7,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
            text = response_text(response)
            # split into interpretations, which are separated by a line break, andremove the prefix "1. ", "2. ", etc.
            interpretations = [re.sub(r"^\d+\. ", "", interpretation) for interpretation in text.split("\n") if interpretation != ""]
            logger.info("semantic interpretations: {}".format(interpretations))
            return interpretations
        except Exception as e:
            print(e)
            print(response)
            print(response_text(response))

if __name__ == "__main__":
    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

    parser = argparse.ArgumentParser()
    # parser.add_argument("--udf_class", type=str, help="UDF class name we want to generate")
    # parser.add_argument("--n_obj", type=int, help="number of objects in the UDF signature")
    # parser.add_argument("--udf_description", type=str, help="UDF description")
    # parser.add_argument("--gt_udf_impl", type=str, help="ground truth UDF implementation")
    # parser.add_argument("--udf_generation_name", type=str, help="name of the function that GPT will generate")
    parser.add_argument("--log_name_suffix", type=str, help="log name suffix")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    # parser.add_argument("--stage", type=str, help="stage of the experiment (udf_generation, udf_selection, or all)")

    args = parser.parse_args()
    # gt_udf_impl = args.gt_udf_impl
    # udf_class = args.udf_class
    # n_obj= args.n_obj
    # udf_description = args.udf_description
    # udf_generation_name = "py_{}".format(udf_class)
    log_name_suffix = args.log_name_suffix
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    num_interpretations = args.num_interpretations
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
    qe = QueryExecutor("Obj_clevrer", registered_functions)
    qp = QueryParser()

    # Parse query
    flag = qp.parse(user_query)
    if "yes" in flag:
        qe.run(parsed_program)
    elif "no" in flag:
        # Step 1: propose new UDFs
        # TODO
        # Step 2: generate semantic interpretations
        generate_semantic_interpretation_base_prompt = prompt_config["prompt"]["generate_semantic_interpretation"]
        generate_udfs_base_prompt = prompt_config["prompt"]["generate_udfs"]
        logger.info("Generating {} UDF interpretations".format(num_interpretations))
        generate_semantic_interpretation_prompt = replace_slot(generate_semantic_interpretation_base_prompt, {
            "num_interpretations": num_interpretations,
            "udf_description": udf_description,
            "obj_dictionary_str": "one object dictionary (o1)" if n_obj == 1 else "two object dictionaries (o1, o2)",
            "n_obj": "one object" if n_obj == 1 else "two objects",
            "udf_header": "py_{}({}, **kwargs)".format(udf_class, "o1" if n_obj == 1 else "o1, o2"),
        })
        interpretations = generate_semantic_interpretation(generate_semantic_interpretation_prompt)
        logger.info("Generating UDFs")

        # Step 3: generate UDFs
        for i in range(num_interpretations):
            logger.info("Generating UDF {}".format(i))
            generate_udfs_prompt = replace_slot(generate_udfs_base_prompt, {
                "udf_class": udf_class,
                "udf_description": udf_description,
                "semantic_interpretation": interpretations[i],
                "n_obj": "one object" if n_obj == 1 else "two objects",
                "udf_header": "py_{}({}, **kwargs)".format(udf_class, "o1" if n_obj == 1 else "o1, o2"),
            })
            generate_udf(i, generate_udfs_prompt, interpretations[i])
    else:
        raise ValueError("Invalid response from user_proxy")
