from typing_extensions import Annotated
import autogen
from vocaludf.parser import parse
from vocaludf.utils import replace_slot
import pyparsing as pp
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
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