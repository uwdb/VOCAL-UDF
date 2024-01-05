import random

GQA_CURATED_EXAMPLES=[
"""Question: Is there a yellow cube in the video?
Program:
BOX0=LOC(video=VIDEO,object='yellow cube')
ANSWER0=COUNT(box=BOX0)
ANSWER1=EVAL(expr="'yes' if any(value > 0 for value in {ANSWER0}.values()) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Is there a red cube on the left of a blue cylinder in the video?
Program:
BOX0=LOC(video=VIDEO,object='blue cylinder')
VIDEO0=CROP_LEFTOF(video=VIDEO,box=BOX0)
BOX1=LOC(video=VIDEO0,object='red cube')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if any(value > 0 for value in {ANSWER0}.values()) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Is there a red cube that is on the left of a blue cylinder, then moves to the right of the blue cylinder in the video?
Program:
BOX0=LOC(video=VIDEO,object='blue cylinder')
VIDEO0=CROP_LEFTOF(video=VIDEO,box=BOX0)
BOX1=LOC(video=VIDEO0,object='red cube')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="[key for key, value in {ANSWER0}.items() if value > 0]")
VIDEO1=CROP_RIGHTOF(video=VIDEO,box=BOX0)
BOX2=LOC(video=VIDEO1,object='red cube')
ANSWER2=COUNT(box=BOX2)
ANSWER3=EVAL(expr="[key for key, value in {ANSWER2}.items() if value > 0]")
ANSWER4=BEFORE(frames1=ANSWER1,frames2=ANSWER3)
FINAL_RESULT=RESULT(var=ANSWER4)
""",
]

def create_prompt(inputs,num_prompts=8,method='random',seed=42,group=0):
    if method=='all':
        prompt_examples = GQA_CURATED_EXAMPLES
    elif method=='random':
        random.seed(seed)
        prompt_examples = random.sample(GQA_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f'Think step by step to answer the question.\n\n{prompt_examples}'


    return prompt_examples + "\nQuestion: {question}\nProgram:".format(**inputs)