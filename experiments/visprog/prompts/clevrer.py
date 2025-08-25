import random

CLEVRER_CURATED_EXAMPLES=[
"""Question: An object o0 is in front of another object o1, and o0 is also at the top of the video frame.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=TOP(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1],min_duration=1)
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # FrontOf(o0, o1), Top(o0)
"""Question: For at least 15 frames, an object o0 is in front of another object o1 and is on the left side of the video frame. Following this, for at least 5 frames, o0 is still on the left side and at the top of the video frame, while another object o1 is to the left of o0.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=LEFT(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1],min_duration=15)
PRED2=TOP(object=OBJT,var='o0')
PRED3=LEFTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o0')
EVENT1=EVENT(predicates=[PRED1,PRED2,PRED3],min_duration=5)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
ANSWER0=EVAL(expr="'yes' if len({EVENT2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # Duration((FrontOf(o0, o1), Left(o0)), 15); Duration((Left(o0), LeftOf(o1, o0), Top(o0)), 5)
"""Question: An object o0 is in front of another object o1, o0 is also at the top of the video frame, and o1 is gray and on the left side of the video frame.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=TOP(object=OBJT,var='o0')
PRED2=GRAY(object=OBJT,var='o1')
PRED3=LEFT(object=OBJT,var='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3],min_duration=1)
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # FrontOf(o0, o1), Top(o0), Color(o1, 'gray'), Left(o1)
"""Question: A red object o0 is at the top of the video frame and to the left of another object o1. Additionally, o0 is also on the left side of the video frame.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=RED(object=OBJT,var='o0')
PRED1=TOP(object=OBJT,var='o0')
PRED2=LEFT(object=OBJT,var='o0')
PRED3=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3],min_duration=1)
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # Top(o0), Color(o0, 'red'), LeftOf(o0, o1), Left(o0))
"""Question: A blue object o0 is on the left side of the video frame, to the left of another object o1, and o1 is also on the left side of the video frame.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=BLUE(object=OBJT,var='o0')
PRED1=LEFT(object=OBJT,var='o0')
PRED2=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED3=LEFT(object=OBJT,var='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3],min_duration=1)
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # Color(o0, 'blue'), Left(o0), LeftOf(o0, o1), Left(o1)
"""Question: An object o0 is in front of o1, o1 is in front of o2, and o0 is to the left of o2. Then, o0 is blue and o2 is at the top of the video frame. Finally, o0, which is made of rubber, is in front of o1.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=FRONTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o2')
PRED2=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2],min_duration=1)
PRED3=BLUE(object=OBJT,var='o0')
PRED4=TOP(object=OBJT,var='o2')
EVENT1=EVENT(predicates=[PRED3,PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED5=RUBBER(object=OBJT,var='o0')
EVENT3=EVENT(predicates=[PRED0,PRED5],min_duration=1)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (FrontOf(o0, o1), FrontOf(o1, o2), LeftOf(o0, o2)); (Color(o0, 'blue'), Top(o2)); (FrontOf(o0, o1), Material(o0, 'rubber'))
"""Question: An object o0 is at the top of the video frame and to the left of another object o1. Later on, object o1, which is made of rubber, is in front of a third object o2, and o2 is shaped like a cube. After this, o0 is to the left of both o1 and o2.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=TOP(object=OBJT,var='o0')
PRED1=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1],min_duration=1)
PRED2=RUBBER(object=OBJT,var='o1')
PRED3=FRONTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o2')
PRED4=CUBE(object=OBJT,var='o2')
EVENT1=EVENT(predicates=[PRED2,PRED3,PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED6=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED7=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
EVENT3=EVENT(predicates=[PRED6,PRED7],min_duration=1)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (Top(o0), LeftOf(o0, o1)); (FrontOf(o1, o2), Material(o1, 'rubber'), Shape(o2, 'cube')); (LeftOf(o0, o1), LeftOf(o0, o2))
"""Question: Initially, an object o0 is at the top of the video frame. Then, another object o1 is also at the top, while o0 is in front of a third object o2, which is made of rubber. Then, o2 is red, o1 is on the left side of the video frame, and o0 remains at the top.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=TOP(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0],min_duration=1)
PRED1=TOP(object=OBJT,var='o1')
PRED2=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED3=RUBBER(object=OBJT,var='o2')
EVENT1=EVENT(predicates=[PRED1,PRED2,PRED3],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED4=RED(object=OBJT,var='o2')
PRED5=LEFT(object=OBJT,var='o1')
PRED6=TOP(object=OBJT,var='o0')
EVENT3=EVENT(predicates=[PRED4,PRED5,PRED6],min_duration=1)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # Top(o0); (Top(o1), FrontOf(o0, o2), Material(o2, 'rubber')); (Color(o2, 'red'), Left(o1), Top(o0))
"""Question: In the first part of the video, an object o0 is to the left of another rubber object o1, and o0 is also in front of a third cube object o2. Then, o0 is again to the left of o1 and on the left side of the video frame. Finally, o1 is to the left of o2.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED2=RUBBER(object=OBJT,var='o1')
PRED3=CUBE(object=OBJT,var='o2')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3],min_duration=1)
PRED4=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED5=LEFT(object=OBJT,var='o0')
EVENT1=EVENT(predicates=[PRED4,PRED5],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED5=LEFTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o2')
EVENT3=EVENT(predicates=[PRED5],min_duration=1)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (LeftOf(o0, o1), Material(o1, 'rubber'), FrontOf(o0, o2), Shape(o2, 'cube')); (LeftOf(o0, o1), Left(o0)); LeftOf(o1, o2)
"""Question: A gray object o0 is to the left of another object o1, then o0 is at the top of the video frame and to the left of a sphere o2, then o1 is at the top of the frame and to the left of o2.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=GRAY(object=OBJT,var='o0')
PRED1=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1],min_duration=1)
PRED2=TOP(object=OBJT,var='o0')
PRED3=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED4=SPHERE(object=OBJT,var='o2')
EVENT1=EVENT(predicates=[PRED2,PRED3,PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED5=TOP(object=OBJT,var='o1')
PRED6=LEFTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o2')
EVENT3=EVENT(predicates=[PRED5,PRED6],min_duration=1)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (Color(o0, 'gray'), LeftOf(o0, o1)); (Top(o0), LeftOf(o0, o2), Shape(o2, 'sphere')); (Top(o1), LeftOf(o1, o2))
"""Question: Initially, an object o0 is in front of and to the left of another object o1, while an object o2 is at the top of the frame. Then, o0 moves to the left of o2, and o2 is red. Finally, for at least 15 frames, o0 is in front of o1, which is made of rubber.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED2=TOP(object=OBJT,var='o2')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2],min_duration=1)
PRED3=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
PRED4=RED(object=OBJT,var='o2')
EVENT1=EVENT(predicates=[PRED3,PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED5=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED6=RUBBER(object=OBJT,var='o1')
EVENT3=EVENT(predicates=[PRED5,PRED6],min_duration=15)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (FrontOf(o0, o1), LeftOf(o0, o1), Top(o2)); (Color(o2, 'red'), LeftOf(o0, o2)); Duration((FrontOf(o0, o1), Material(o1, 'rubber')), 15)
"""Question: For at least 5 frames, an object o0 is to the left of another object o1, and both o2 and o1 are on the left side of the video frame. Then, o2 is of material rubber and o0 is in front of o1 for at least 15 frames. Finally, o2 is red and o0 is in front of o1.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=LEFT(object=OBJT,var='o2')
PRED2=LEFT(object=OBJT,var='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2],min_duration=5)
PRED3=RUBBER(object=OBJT,var='o2')
PRED4=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT1=EVENT(predicates=[PRED3,PRED4],min_duration=15)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED5=RED(object=OBJT,var='o2')
PRED6=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT3=EVENT(predicates=[PRED5,PRED6],min_duration=1)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # Duration((LeftOf(o0, o1), Left(o2), Left(o1)), 5); Duration((Material(o2, 'rubber'), FrontOf(o0, o1)), 15); (Color(o2, 'red'), FrontOf(o0, o1))
"""Question: Initially, an object o0 is on the left side of the video frame for at least 10 frames. Then, for at least 5 frames, o1 is to the left of o2 and o0 is to the left of o1. In a subsequent sequence lasting at least 10 frames, o0 is red, o2 is on the left side of the frame and made of rubber, and o1 is to the left of o2.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=LEFT(object=OBJT,var='o0')
EVENT0=EVENT(predicates=[PRED0],min_duration=10)
PRED1=LEFTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o2')
PRED2=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT1=EVENT(predicates=[PRED1,PRED2],min_duration=5)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED3=RED(object=OBJT,var='o0')
PRED4=LEFT(object=OBJT,var='o2')
PRED5=RUBBER(object=OBJT,var='o2')
PRED6=LEFTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o2')
EVENT3=EVENT(predicates=[PRED3,PRED4,PRED5,PRED6],min_duration=10)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # Duration(Left(o0), 10); Duration((LeftOf(o1, o2), LeftOf(o0, o1)), 5); Duration((Color(o0, 'red'), Left(o2), Material(o2, 'rubber'), LeftOf(o1, o2)), 10)
"""Question: For at least 10 frames, an object o0 is in front of another object o1, which is at the top of the video frame. In the next segment lasting at least 5 frames, o0 is in front of a third object o2. Finally, for another duration of at least 5 frames, o1 is in front of o2, o0 is blue, o1 is shaped like a cube, and o1 remains at the top of the frame.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=TOP(object=OBJT,var='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1],min_duration=10)
PRED2=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o2')
EVENT1=EVENT(predicates=[PRED2],min_duration=5)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED3=FRONTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o2')
PRED4=BLUE(object=OBJT,var='o0')
PRED5=CUBE(object=OBJT,var='o1')
PRED6=TOP(object=OBJT,var='o1')
EVENT3=EVENT(predicates=[PRED3,PRED4,PRED5,PRED6],min_duration=5)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # Duration((FrontOf(o0, o1), Top(o1)), 10); Duration(FrontOf(o0, o2), 5); Duration((FrontOf(o1, o2), Color(o0, 'blue'), Shape(o1, 'cube'), Top(o1)), 5)
"""Question: For at least 10 frames, a sphere o0 is in front of another green object o1. Then, lasting at least 5 frames, o1 is in front of a third object o2, o0 is to the left of o1. Finally, for a duration of at least 15 frames, o0 is in front of o1, and o1 is on the left side of the video frame.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED1=GREEN(object=OBJT,var='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1],min_duration=10)
PRED2=FRONTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o2')
PRED3=LEFTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED4=SPHERE(object=OBJT,var='o0')
EVENT1=EVENT(predicates=[PRED2,PRED3,PRED4],min_duration=5)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED5=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
PRED6=LEFT(object=OBJT,var='o1')
EVENT3=EVENT(predicates=[PRED5,PRED6],min_duration=15)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # Duration((FrontOf(o0, o1), Color(o1, 'green')), 10); Duration((FrontOf(o1, o2), LeftOf(o0, o1), Shape(o0, 'sphere')), 5); Duration((FrontOf(o0, o1), Left(o1)), 15)
"""Question: A gray cube o1 is to the left of an object o2, and further to the left of o1 is a gray sphere o3.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=GRAY(object=OBJT,var='o1')
PRED1=CUBE(object=OBJT,var='o1')
PRED2=LEFTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o2')
PRED3=GRAY(object=OBJT,var='o3')
PRED4=SPHERE(object=OBJT,var='o3')
PRED5=LEFTOF(object1=OBJT,var1='o2',object2=OBJT,var2='o3')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4,PRED5],min_duration=1)
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # Color(o1, 'gray'), Shape(o1, 'cube'), Color(o3, 'gray'), Shape(o3, 'sphere'), LeftOf(o1, o2), LeftOf(o2, o3)
"""Question: A red rubber object o0 is in front of another object o1 that is in the top of the video, then o1 moves to the front of o0, and then a third object o2 is in the top left of the video for at least 25 frames.
Program:
```python
OBJ0=LOC(video=VIDEO,object='object')
OBJT=TRACK(object=OBJ0)
PRED0=RED(object=OBJT,var='o0')
PRED1=RUBBER(object=OBJT,var='o0')
PRED2=TOP(object=OBJT,var='o1')
PRED3=FRONTOF(object1=OBJT,var1='o0',object2=OBJT,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3],min_duration=1)
PRED4=FRONTOF(object1=OBJT,var1='o1',object2=OBJT,var2='o0')
EVENT1=EVENT(predicates=[PRED4],min_duration=1)
EVENT2=BEFORE(event1=EVENT0, event2=EVENT1)
PRED5=TOP(object=OBJT,var='o2')
PRED6=LEFT(object=OBJT,var='o2')
EVENT3=EVENT(predicates=[PRED5,PRED6],min_duration=25)
EVENT4=BEFORE(event1=EVENT2, event2=EVENT3)
ANSWER0=EVAL(expr="'yes' if len({EVENT4}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
```
""", # (Color(o0, 'red'), Material(o0, 'rubber'), FrontOf(o0, o1), Top(o1)); FrontOf(o1, o0); Duration((Top(o2), Left(o2)), 25)
]

def create_prompt(inputs,num_prompts=8,method='random',seed=42,group=0,prompt_modules=None):
    if method=='all':
        prompt_examples = CLEVRER_CURATED_EXAMPLES
    elif method=='random':
        random.seed(seed)
        prompt_examples = random.sample(CLEVRER_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    if prompt_modules is not None:
        prompt_examples = f'{prompt_modules}\n\n{prompt_examples}'
    prompt_examples = f'Think step by step to answer the question.\n\n{prompt_examples}'


    return prompt_examples + "\nQuestion: {question}\nProgram:".format(**inputs)