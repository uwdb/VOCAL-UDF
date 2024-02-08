from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import autogen
from src.utils import program_to_dsl, dsl_to_program
import yaml
import json
from parser import parse
import pyparsing as pp

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
    },
)

def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        new_text = text.replace("{{" + key +"}}", value.replace('"', "'"))
    return new_text

config = yaml.load(open("prompt.yaml", "r"), Loader=yaml.FullLoader)

registered_functions = json.load(open("registered_udfs.json", "r"))

# Parse query
system_message = replace_slot(
    config["system_prompt"]["parse_query"],
    {"functions": "\n".join(["{}: {}".format(key, value) for key, value in registered_functions.items()])}
)


chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message=system_message,
    llm_config={
        "config_list": config_list,
        "timeout": 120,
        "temperature": 0,
    }
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="Verify syntax correctness of input query.")
def verify_syntax(
    query: Annotated[str, "Input query written in DSL to be verified."],
) -> str:
    try:
        result = parse().parseString(query, parseAll=True).as_dict()
        # results_dump = parse().parseString(s, parseAll=True).dump()
    except (pp.ParseException, pp.ParseSyntaxException) as err:
        return err.explain()
    except Exception as e:
        error_message = query + "failed:\n" + str(e)
        return error_message
    else:
        return result

user_proxy.initiate_chat(
    chatbot,
    # message="Two objects move from far to close, then to far again",
    message="For at least 15 frames, an object o2 is to the left of a red object o0, and another object o1 is in the front of o2. Then, for a minimum of 5 frames, o1 is far from o2 and o0 is to the left of o2. Finally, o2 is behind o0, and o1 is made of metal.",
)