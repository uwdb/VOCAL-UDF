import random

VAW_CURATED_EXAMPLES=[
"""Question: An object o0 is above another object o1 and is also behind a third object o2 which is orange.
Program:
```python
OBJ0=LOC(image=IMAGE,object='object')
PRED0=ABOVE(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
PRED1=BEHIND(object1=OBJ0,var1='o0',object2=OBJ0,var2='o2')
PRED2=ORANGE(object=OBJ0,var='o2')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (above(o0, o1), behind(o0, o2), orange(o2))
"""Question: An object o0 is beneath another object o1, and there is also an object o2 whose size is small.
Program:
```python
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BENEATH(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
PRED1=SMALL(object=OBJ0,var='o2')
EVENT0=EVENT(predicates=[PRED0,PRED1])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (beneath(o0, o1), small(o2))
"""Question: An object o0 is beneath and behind another object o1, and there is also an object o2 whose length is long.
Program:
```python
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BENEATH(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
PRED1=BEHIND(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
PRED2=LONG(object=OBJ0,var='o2')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
```
""", # (beneath(o0, o1), behind(o0, o1), long(o2))
"""Question: An object o0 is beneath another object o1, o0 is red, and o1 is behind o0.
Program:
```python
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BENEATH(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
PRED1=RED(object=OBJ0,var='o0')
PRED2=BEHIND(object1=OBJ0,var1='o1',object2=OBJ0,var2='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (beneath(o0, o1), red(o0), behind(o1, o0))
"""Question: An object o1 is above a small, brown object o0.
Program:
```python
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BROWN(object=OBJ0,var='o0')
PRED1=SMALL(object=OBJ0,var='o0')
PRED2=ABOVE(object1=OBJ0,var1='o1',object2=OBJ0,var2='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (brown(o0), small(o0), above(o1, o0))
]

def create_prompt(inputs,num_prompts=8,method='random',seed=42,group=0,prompt_modules=None):
    if method=='all':
        prompt_examples = VAW_CURATED_EXAMPLES
    elif method=='random':
        random.seed(seed)
        if num_prompts > len(VAW_CURATED_EXAMPLES):
            prompt_examples = VAW_CURATED_EXAMPLES
        else:
            prompt_examples = random.sample(VAW_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    if prompt_modules is not None:
        prompt_examples = f'{prompt_modules}\n\n{prompt_examples}'
    prompt_examples = f'Think step by step to answer the question.\n\n{prompt_examples}'


    return prompt_examples + "\nQuestion: {question}\nProgram:".format(**inputs)