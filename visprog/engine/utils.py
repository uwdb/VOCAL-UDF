import os
from PIL import Image
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import numpy as np
import copy
import re
from .step_interpreters import register_step_interpreters, parse_step

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

class Program:
    def __init__(self,prog_str,init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        instructions = self.prog_str.split('\n')
        # Remove comments, empty lines, and leading/trailing spaces
        cleaned_instructions = []
        for instruction in instructions:
            if instruction.startswith('#'):
                continue
            comment_idx = instruction.find('#')
            if comment_idx!=-1:
                instruction = instruction[:comment_idx]
            instruction = instruction.strip()
            if instruction=='':
                continue
            cleaned_instructions.append(instruction)
        self.instructions = cleaned_instructions


class ProgramInterpreter:
    def __init__(self,dataset='nlvr', use_precomputed=False, module_list=None):
        self.step_interpreters = register_step_interpreters(dataset, use_precomputed, module_list)

    def execute_step(self,prog_step,inspect):
        step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        print(step_name)
        return self.step_interpreters[step_name].execute(prog_step,inspect)

    def execute(self,prog,init_state,inspect=False):
        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions] # init_state is shared across all steps

        html_str = '<hr>'
        for prog_step in prog_steps:
            # print("prog_step.state: ", prog_step.state)
            if inspect:
                step_output, step_html = self.execute_step(prog_step,inspect)
                html_str += step_html + '<hr>'
            else:
                step_output = self.execute_step(prog_step,inspect)

        if inspect:
            return step_output, prog.state, html_str

        return step_output, prog.state


class ProgramGenerator():
    def __init__(self,prompter,temperature=0.2,top_p=0.5,prob_agg='mean', llm_model="gpt-3.5-turbo-instruct"):
        self.prompter = prompter
        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg
        self.llm_model = llm_model

    def compute_prob(self,response):
        eos = '<|endoftext|>'
        for i,token in enumerate(response.choices[0].logprobs.tokens):
            if token==eos:
                break

        if self.prob_agg=='mean':
            agg_fn = np.mean
        elif self.prob_agg=='sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0].logprobs.token_logprobs[:i]))

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate(self,inputs,retry=0):
        print(self.prompter(inputs))
        if self.llm_model == "gpt-3.5-turbo-instruct":
            print("Using gpt-3.5-turbo-instruct")
            response = client.completions.create(
                model=self.llm_model,
                prompt=self.prompter(inputs),
                temperature=self.temperature,
                max_tokens=512,
                top_p=self.top_p,
                logprobs=1,
                seed=42+retry
            )

            prob = self.compute_prob(response)
            prog = response.choices[0].text.lstrip('\n').rstrip('\n')
        elif self.llm_model in ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]:
            print("Using ", self.llm_model)
            system_prompt = "Generate a program based on the question, with no other comments, inline comments, syntax highlighter, explanations, reasoning, or dialogue."
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.prompter(inputs)},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=512,
                seed=42+retry
            )

            prob = None
            prog = response.choices[0].message.content.lstrip('\n').rstrip('\n')
            # remove duplicate newlines
            prog = re.sub(r'\n+', '\n', prog)

        return prog, prob
