from typing_extensions import Annotated
import autogen
from vocaludf.parser import parse
from vocaludf.utils import replace_slot
import pyparsing as pp
import logging
import os

def behind(o1_y1, o1_y2, o2_y1, o2_y2):
    o1_center_y = (o1_y1 + o1_y2) / 2
    o2_center_y = (o2_y1 + o2_y2) / 2
    return o1_center_y < o2_center_y

def behind(o1_y1, o1_y2, o2_y1, o2_y2, **kwargs):
    threshold = kwargs.get('threshold', 50)
    o1_cy = (o1_y1 + o1_y2) / 2
    o2_cy = (o2_y1 + o2_y2) / 2
    return o2_cy - o1_cy > threshold

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
class QueryParser:
    def __init__(
        self,
        config,
        prompt_config,
        dataset,
        registered_functions,
        object_domain,
        run_id,
        allow_new_udfs=True,
    ):
        self.config = config
        self.run_id = run_id
        self.allow_new_udfs = allow_new_udfs
        self.registered_function_names = []
        for registered_function in registered_functions:
            self.registered_function_names.append(
                registered_function["signature"].split("(")[0].lower()
            )
        self.object_domain = object_domain
        if dataset in ["clevrer", "charades", "cityflow"]:  # video dataset
            dsl_definition_prompt = prompt_config["dsl_definition"]
        elif dataset in ["clevr", "gqa", "vaw"]:  # image dataset
            dsl_definition_prompt = prompt_config["dsl_definition_image"]
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        if allow_new_udfs:
            self.system_message = replace_slot(
                " ".join(
                    [
                        dsl_definition_prompt,
                        prompt_config["udf_definition"]["without_object"] if dataset in ["clevr", "clevrer", "vaw", "cityflow"] else prompt_config["udf_definition"]["with_object"],
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
            self.is_termination_msg = lambda x: x.get("content", "") and (
                "parse_yes" in x.get("content", "").rstrip().lower()
                or "parse_no" in x.get("content", "").rstrip().lower()
            )
        else:
            self.system_message = replace_slot(
                " ".join(
                    [
                        dsl_definition_prompt,
                        prompt_config["udf_definition"]["without_object"] if dataset in ["clevr", "clevrer", "vaw", "cityflow"] else prompt_config["udf_definition"]["with_object"],
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
            self.is_termination_msg = (
                lambda x: x.get("content", "")
                and "terminate" in x.get("content", "").rstrip().lower()
            )
        logger.debug("system_message: {}".format(self.system_message))

    def parse(self, user_query):
        def verify_syntax(
                query: Annotated[str, "Input query written in DSL to be verified."]
            ) -> str:
                def check_UDF_support(program):
                    unsupported_udfs = []
                    unsupported_objects = []
                    flag = True
                    for sg in program["query"]:
                        for pred in sg["scene_graph"]:
                            if pred["predicate"].lower() not in self.registered_function_names:
                                flag = False
                                unsupported_udfs.append(pred["predicate"])
                            if pred["predicate"].lower() == "object" and pred["parameter"].lower() not in self.object_domain:
                                flag = False
                                unsupported_objects.append(pred["parameter"])
                    return flag, unsupported_udfs, unsupported_objects

                try:
                    parsed_program = parse().parseString(query, parseAll=True).as_dict()
                    # Post-check if parsed program uses unsupported UDFs
                    flag, unsupported_udfs, unsupported_objects = check_UDF_support(parsed_program)
                    if flag:
                        self.parsed_program = parsed_program
                        self.parsed_query = query
                        return self.parsed_program
                    else:
                        return (
                            query
                            + " failed:\n"
                            + "Unsupported UDFs: {}\n".format(unsupported_udfs) if unsupported_udfs else ""
                            + "Unsupported Objects: {}\n".format(unsupported_objects) if unsupported_objects else ""
                        )
                except (pp.ParseException, pp.ParseSyntaxException) as err:
                    return err.explain()
                except Exception as e:
                    error_message = query + " failed:\n" + str(e)
                    return error_message

        for trial in range(3): # retry 3 times
            logger.info("Trial {}".format(trial))
            self.parser = autogen.AssistantAgent(
                name="parser",
                system_message=self.system_message,
                llm_config={
                    "config_list": [{
                        'model': 'gpt-4-turbo-2024-04-09',
                        'api_key': os.getenv("OPENAI_API_KEY"),
                    }],
                    "timeout": 120,
                    "temperature": self.config["query_parser"]["temperature"],
                    "seed": self.run_id * 42 + trial,
                    "cache_seed": None,
                },
            )

            self.user_proxy = autogen.UserProxyAgent(
                name="user_proxy",
                is_termination_msg=self.is_termination_msg,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=5,
                code_execution_config=False,
                # code_execution_config={"work_dir": "coding", "use_docker": False},
            )

            self.parser.register_for_llm(name="verify_syntax", description="Verify syntax correctness of input query.")(verify_syntax)
            self.user_proxy.register_for_execution(name="verify_syntax")(verify_syntax)

            self.user_proxy.initiate_chat(
                self.parser,
                message=user_query,
            )
            chat_messages = self.user_proxy.chat_messages[self.parser]
            logger.debug("chat_messages {}".format(chat_messages))
            if chat_messages[-1]["role"] == "user":
                flag = chat_messages[-1]["content"].strip().lower()
                logger.debug("flag {}".format(flag))
                if self.allow_new_udfs and ("parse_yes" in flag or "parse_no" in flag):
                    return flag
                elif not self.allow_new_udfs and "terminate" in flag:
                    return flag
        # The conversation didn't end with the user's message (YES/NO)
        # Assume NO
        flag = "parse_no"
        logger.debug(
            "The conversation didn't end with the user's message. Assume: flag {}".format(
                flag
            )
        )
        return flag

    def get_parsed_program(self):
        return self.parsed_program

    def get_parsed_query(self):
        return self.parsed_query