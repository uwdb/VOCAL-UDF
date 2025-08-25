import random

CHARADES_CURATED_EXAMPLES=[
"""Question: A person, who is touching object o1, is leaning on and positioned above it. Additionally, there is an object o2 classified as food.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=ABOVE(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED2=PERSON(object=OBJT,var='o0')
PRED3=LEANINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED4=FOOD(object=OBJT,var='o2')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4],min_duration=1)
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (above(o0, o1), touching(o0, o1), object(o0, 'person'), leaning_on(o0, o1)); object(o2, 'food')
"""Question: A person is writing on object o1. Later, the person is touching o1. Finally, the person is wearing some clothes.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=WRITINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=PERSON(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1],min_duration=1)
PRED2=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT1=EVENT(predicates=[PRED2],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED3=WEARING(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED4=CLOTHES(object=OBJT,var='o2')
EVENT3=EVENT(predicates=[PRED3,PRED4],min_duration=1)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (writing_on(o0, o1), object(o0, 'person'), object(o2, 'clothes')); touching(o0, o1); wearing(o0, o2)
"""Question: A person is drinking from a cup, glass, or bottle, lying on another object o2, while simultaneously touching o2.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=DRINKINGFROM(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=LYINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED2=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED3=CUPGLASSBOTTLE(object=OBJT,var='o0')
PRED4=PERSON(object=OBJT,var='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4],min_duration=1)
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # object(o0, 'cup/glass/bottle'); (drinking_from(o1, o0), lying_on(o1, o2), touching(o1, o2)); object(o1, 'person')
"""Question: Initially, a person is touching object o1. Later, the person is wearing both object o1 and another shoe o2.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=PERSON(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1],min_duration=1)
PRED2=WEARING(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED3=SHOE(object=OBJT,var='o2')
PRED4=WEARING(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
EVENT1=EVENT(predicates=[PRED2,PRED3,PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
ANSWER0=EVAL(expr="'yes' if len({EVENT2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # touching(o0, o1); (wearing(o0, o2), wearing(o0, o1), object(o0, 'person')); object(o2, 'shoe')
"""Question: The person is leaning on the chair, and leaning on another object while touching it.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=LEANINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=LEANINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED2=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED3=CHAIR(object=OBJT,var='o1')
PRED4=PERSON(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4],min_duration=1)
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # object(o0, 'chair'); (leaning_on(o1, o0), leaning_on(o1, o2), object(o1, 'person'), touching(o1, o2))
"""Question: A person is above an object o1 and is also touching another object o2. Later, the person moves to be in front of o2 and is twisting o2.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=ABOVE(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED2=PERSON(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2],min_duration=1)
PRED3=INFRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED4=TWISTING(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
EVENT1=EVENT(predicates=[PRED3,PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
ANSWER0=EVAL(expr="'yes' if len({EVENT2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (above(o0, o1), touching(o0, o2)); object(o0, 'person'); (in_front_of(o0, o2), twisting(o0, o2))
"""Question: Initially, a person is touching an object o1 and is in front of another object o2. Subsequently, this person is touching o2, and finally, the person is above o1.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=INFRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED2=PERSON(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2],min_duration=1)
PRED3=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
EVENT1=EVENT(predicates=[PRED3],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED4=ABOVE(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT3=EVENT(predicates=[PRED4],min_duration=1)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (touching(o0, o1), in_front_of(o0, o2)); (touching(o0, o2), object(o0, 'person')); above(o0, o1)
"""Question: A person is leaning on an object o1. Later, the person is touching it. Later, the person is in front of another object o2 while touching it.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=LEANINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT0=EVENT(predicates=[PRED0],min_duration=1)
PRED1=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT1=EVENT(predicates=[PRED1],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED2=INFRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED3=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED4=PERSON(object=OBJT,var='o0')
EVENT3=EVENT(predicates=[PRED2,PRED3,PRED4],min_duration=1)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # leaning_on(o0, o1); touching(o0, o1); (object(o0, 'person'), in_front_of(o0, o2), touching(o0, o2))
"""Question: A person is in front of an object o1 and drinking another object o2. Later in the sequence, the person is above o2 and wearing o1.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=INFRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=DRINKINGFROM(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED2=PERSON(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2],min_duration=1)
PRED3=ABOVE(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED4=WEARING(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT1=EVENT(predicates=[PRED3,PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
ANSWER0=EVAL(expr="'yes' if len({EVENT2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (in_front_of(o0, o1), drinking_from(o0, o2)); object(o0, 'person'); (above(o0, o2), wearing(o0, o1))
"""Question: A person is lying on and touching an object o1, and positioned above another object o2. In a subsequent scene, the person is writing on o1.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=LYINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED2=ABOVE(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED3=PERSON(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3],min_duration=1)
PRED4=WRITINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT1=EVENT(predicates=[PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
ANSWER0=EVAL(expr="'yes' if len({EVENT2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (lying_on(o0, o1), above(o0, o2), touching(o0, o1)); object(o0, 'person'); writing_on(o0, o1)
"""Question: A person is involved in a sequence where they are writing on a paper/notebook. Subsequently, the person is touching another object o2 and is twisting it.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=WRITINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=PERSON(object=OBJT,var='o0')
PRED2=PAPERNOTEBOOK(object=OBJT,var='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2],min_duration=1)
PRED3=TOUCHING(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED4=TWISTING(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
EVENT1=EVENT(predicates=[PRED3,PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
ANSWER0=EVAL(expr="'yes' if len({EVENT2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # object(o0, 'person'); (writing_on(o0, o1), object(o1, 'paper/notebook')); (touching(o0, o2), twisting(o0, o2))
"""Question: A person is drinking object o1 while sitting on another object o2. Following this, the person is leaning on o2 and in front of o2.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=DRINKINGFROM(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=SITTINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED2=PERSON(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2],min_duration=1)
PRED3=LEANINGON(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED4=INFRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
EVENT1=EVENT(predicates=[PRED3,PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
ANSWER0=EVAL(expr="'yes' if len({EVENT2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (drinking_from(o0, o1), object(o0, 'person'), sitting_on(o0, o2)); (leaning_on(o0, o2), in_front_of(o0, o2))
]

def create_prompt(inputs,num_prompts=8,method='random',seed=42,group=0,prompt_modules=None):
    if method=='all':
        prompt_examples = CHARADES_CURATED_EXAMPLES
    elif method=='random':
        random.seed(seed)
        if num_prompts > len(CHARADES_CURATED_EXAMPLES):
            prompt_examples = CHARADES_CURATED_EXAMPLES
        else:
            prompt_examples = random.sample(CHARADES_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    if prompt_modules is not None:
        prompt_examples = f'{prompt_modules}\n\n{prompt_examples}'
    prompt_examples = f'Think step by step to answer the question.\n\n{prompt_examples}'


    return prompt_examples + "\nQuestion: {question}\nProgram:".format(**inputs)