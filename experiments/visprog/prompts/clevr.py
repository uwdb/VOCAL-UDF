import random

CLEVR_CURATED_EXAMPLES=[
"""Question: a gray cube is left of a big object, which is left of a large gray sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BIG(object=OBJ0,var='o1')
PRED1=GRAY(object=OBJ0,var='o2')
PRED2=CUBE(object=OBJ0,var='o2')
PRED3=BIG(object=OBJ0,var='o3')
PRED4=GRAY(object=OBJ0,var='o3')
PRED5=SPHERE(object=OBJ0,var='o3')
PRED6=LEFTOF(object1=OBJ0,var1='o2',object2=OBJ0,var2='o1')
PRED7=LEFTOF(object1=OBJ0,var1='o1',object2=OBJ0,var1='o3')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4,PRED5,PRED6,PRED7])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Color(o1, 'gray'), Shape(o1, 'cube'), Size(o2, 'big'), Color(o2, 'gray'), Shape(o2, 'sphere'), LeftOf(o1, o0), LeftOf(o0, o2)
"""Question: a gray rubber sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=GRAY(object=OBJ0,var='o0')
PRED1=RUBBER(object=OBJ0,var='o0')
PRED2=SPHERE(object=OBJ0,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'gray'), Material(o0, 'rubber'), Shape(o0, 'sphere')
"""Question: a big blue block
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BIG(object=OBJ0,var='o0')
PRED1=BLUE(object=OBJ0,var='o0')
PRED2=CUBE(object=OBJ0,var='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
""", # Size(o0, 'big'), Color(o0, 'blue'), Shape(o0, 'cube')
"""Question: a big gray object left of a red object
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=RED(object=OBJ0,var='o0')
PRED1=BIG(object=OBJ0,var='o1')
PRED2=GRAY(object=OBJ0,var='o1')
PRED3=LEFTOF(object1=OBJ0,var1='o1',object2=OBJ0,var2='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'red'), Size(o1, 'big'), Color(o1, 'gray'), LeftOf(o1, o0)
"""Question: a big matte sphere and a matte sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BIG(object=OBJ0,var='o0')
PRED1=RUBBER(object=OBJ0,var='o0')
PRED2=SPHERE(object=OBJ0,var='o0')
PRED3=RUBBER(object=OBJ0,var='o1')
PRED4=SPHERE(object=OBJ0,var='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Material(o0, 'rubber'), Shape(o0, 'sphere'), Material(o1, 'rubber'), Shape(o1, 'sphere')
"""Question: a matte thing is in front of the gray ball, and it is the same size as the gray object.
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=RUBBER(object=OBJ0,var='o0')
PRED1=SPHERE(object=OBJ0,var='o1')
PRED2=GRAY(object=OBJ0,var='o1')
PRED3=FRONTOF(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
PRED4=EQUALSIZE(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Material(o0, 'rubber'), Shape(o1, 'sphere'), Color(o1, 'gray'), FrontOf(o0, o1), EqualSize(o0, o1)
"""Question: a gray rubber sphere is in front of the huge red rubber cube
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=GRAY(object=OBJ0,var='o0')
PRED1=RUBBER(object=OBJ0,var='o0')
PRED2=SPHERE(object=OBJ0,var='o0')
PRED3=BIG(object=OBJ0,var='o1')
PRED4=RED(object=OBJ0,var='o1')
PRED5=RUBBER(object=OBJ0,var='o1')
PRED6=CUBE(object=OBJ0,var='o1')
PRED7=FRONTOF(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4,PRED5,PRED6,PRED7])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'gray'), Material(o0, 'rubber'), Shape(o0, 'sphere'), Size(o1, 'big'), Color(o1, 'red'), Material(o1, 'rubber'), Shape(o1, 'cube'), FrontOf(o0, o1)
"""Question: The red cube and the large green sphere are made of the same material.
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=RED(object=OBJ0,var='o0')
PRED1=CUBE(object=OBJ0,var='o0')
PRED2=BIG(object=OBJ0,var='o1')
PRED3=GREEN(object=OBJ0,var='o1')
PRED4=SPHERE(object=OBJ0,var='o1')
PRED5=EQUALMATERIAL(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4,PRED5])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'red'), Shape(o0, 'cube'), Size(o1, 'big'), Color(o1, 'green'), Shape(o1, 'sphere'), EqualMaterial(o0, o1)
"""Question: A cube is in front of the big rubber sphere and on the left side of the big blue rubber object.
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=CUBE(object=OBJ0,var='o0')
PRED1=BIG(object=OBJ0,var='o1')
PRED2=RUBBER(object=OBJ0,var='o1')
PRED3=SPHERE(object=OBJ0,var='o1')
PRED4=FRONTOF(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
PRED5=BIG(object=OBJ0,var='o2')
PRED6=BLUE(object=OBJ0,var='o2')
PRED7=RUBBER(object=OBJ0,var='o2')
PRED8=LEFTOF(object1=OBJ0,var1='o0',object2=OBJ0,var2='o2')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4,PRED5,PRED6,PRED7,PRED8])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
""", # Shape(o0, 'cube'), FrontOf(o0, o1), Size(o1, 'big'), Material(o1, 'rubber'), Shape(o1, 'sphere'), Size(o2, 'big'), Color(o2, 'blue'), Material(o2, 'rubber'), LeftOf(o0, o2)
"""Question: There is a big matte sphere, which is the same material as the big rubber object in front of the blue cube.
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BIG(object=OBJ0,var='o0')
PRED1=RUBBER(object=OBJ0,var='o0')
PRED2=SPHERE(object=OBJ0,var='o0')
PRED3=BIG(object=OBJ0,var='o1')
PRED4=RUBBER(object=OBJ0,var='o1')
PRED5=EQUALMATERIAL(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
PRED6=BLUE(object=OBJ0,var='o2')
PRED7=CUBE(object=OBJ0,var='o2')
PRED8=FRONTOF(object1=OBJ0,var1='o1',object2=OBJ0,var2='o2')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4,PRED5,PRED6,PRED7,PRED8])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Material(o0, 'rubber'), Shape(o0, 'sphere'), Size(o1, 'big'), Material(o1, 'rubber'), EqualMaterial(o0, o1), Color(o2, 'blue'), Shape(o2, 'cube'), FrontOf(o1, o2)
"""Question: A gray sphere is the same size as a gray matte sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=GRAY(object=OBJ0,var='o0')
PRED1=SPHERE(object=OBJ0,var='o0')
PRED2=GRAY(object=OBJ0,var='o1')
PRED3=RUBBER(object=OBJ0,var='o1')
PRED4=SPHERE(object=OBJ0,var='o1')
PRED5=EQUALSIZE(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4,PRED5])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'gray'), Shape(o0, 'sphere'), Color(o1, 'gray'), Material(o1, 'rubber'), Shape(o1, 'sphere'), EqualSize(o0, o1)
"""Question: A green rubber thing is the same size as a red cube
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=GREEN(object=OBJ0,var='o0')
PRED1=RUBBER(object=OBJ0,var='o0')
PRED2=RED(object=OBJ0,var='o1')
PRED3=CUBE(object=OBJ0,var='o1')
PRED4=EQUALSIZE(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'green'), Material(o0, 'rubber'), Color(o1, 'red'), Shape(o1, 'cube'), EqualSize(o0, o1)
"""Question: A large gray thing is on the left side of the thing that is in front of the large gray ball behind the large gray matte thing
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BIG(object=OBJ0,var='o0')
PRED1=GRAY(object=OBJ0,var='o0')
PRED2=BIG(object=OBJ0,var='o1')
PRED3=GRAY(object=OBJ0,var='o1')
PRED4=SPHERE(object=OBJ0,var='o1')
PRED5=BIG(object=OBJ0,var='o2')
PRED6=GRAY(object=OBJ0,var='o2')
PRED7=RUBBER(object=OBJ0,var='o2')
PRED8=LEFTOF(object1=OBJ0,var1='o0',object2=OBJ0,var2='o3')
PRED9=FRONTOF(object1=OBJ0,var1='o3',object2=OBJ0,var2='o1')
PRED10=FRONTOF(object1=OBJ0,var1='o2',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4,PRED5,PRED6,PRED7,PRED8,PRED9,PRED10])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Color(o0, 'gray'), Size(o1, 'big'), Color(o1, 'gray'), Shape(o1, 'sphere'), Size(o2, 'big'), Color(o2, 'gray'), Material(o2, 'rubber'), LeftOf(o0, o3), FrontOf(o3, o1), FrontOf(o2, o1)
"""Question: A large matte object is in front of a large cube on the right side of a large gray matte ball
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BIG(object=OBJ0,var='o0')
PRED1=RUBBER(object=OBJ0,var='o0')
PRED2=BIG(object=OBJ0,var='o1')
PRED3=CUBE(object=OBJ0,var='o1')
PRED4=BIG(object=OBJ0,var='o2')
PRED5=GRAY(object=OBJ0,var='o2')
PRED6=RUBBER(object=OBJ0,var='o2')
PRED7=SPHERE(object=OBJ0,var='o2')
PRED8=FRONTOF(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
PRED9=LEFTOF(object1=OBJ0,var1='o2',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4,PRED5,PRED6,PRED7,PRED8,PRED9])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Material(o0, 'rubber'), Size(o1, 'big'), Shape(o1, 'cube'), Size(o2, 'big'), Color(o2, 'gray'), Material(o2, 'rubber'), Shape(o2, 'sphere'), FrontOf(o0, o1), LeftOf(o2, o1)
"""Question: A cube is made of the same material as a gray sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=CUBE(object=OBJ0,var='o0')
PRED1=GRAY(object=OBJ0,var='o1')
PRED2=SPHERE(object=OBJ0,var='o1')
PRED3=EQUALMATERIAL(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Shape(o0, 'cube'), Color(o1, 'gray'), Shape(o1, 'sphere'), EqualMaterial(o0, o1)
"""Question: There is a object that is in front of a big green thing
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BIG(object=OBJ0,var='o0')
PRED1=GREEN(object=OBJ0,var='o0')
PRED2=FRONTOF(object1=OBJ0,var1='o1',object2=OBJ0,var2='o0')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Color(o0, 'green'), FrontOf(o1, o0)
"""Question: There is an object that is the same size as the blue object
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BLUE(object=OBJ0,var='o0')
PRED1=EQUALSIZE(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'blue'), EqualSize(o0, o1)
"""Question: A big thing is made of the same material as the gray block.
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BIG(object=OBJ0,var='o0')
PRED1=GRAY(object=OBJ0,var='o1')
PRED2=CUBE(object=OBJ0,var='o1')
PRED3=EQUALMATERIAL(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Color(o1, 'gray'), Shape(o1, 'cube'), EqualMaterial(o0, o1)
"""Question: A large gray thing is to the right of the gray sphere that is in front of the rubber object in front of the large rubber sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=BIG(object=OBJ0,var='o0')
PRED1=GRAY(object=OBJ0,var='o0')
PRED2=GRAY(object=OBJ0,var='o1')
PRED3=SPHERE(object=OBJ0,var='o1')
PRED4=RUBBER(object=OBJ0,var='o2')
PRED5=BIG(object=OBJ0,var='o3')
PRED6=RUBBER(object=OBJ0,var='o3')
PRED7=SPHERE(object=OBJ0,var='o3')
PRED8=LEFTOF(object1=OBJ0,var1='o1',object2=OBJ0,var2='o0')
PRED9=FRONTOF(object1=OBJ0,var1='o1',object2=OBJ0,var2='o2')
PRED10=FRONTOF(object1=OBJ0,var1='o2',object2=OBJ0,var2='o3')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3,PRED4,PRED5,PRED6,PRED7,PRED8,PRED9,PRED10])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Color(o0, 'gray'), Color(o1, 'gray'), Shape(o1, 'sphere'), Material(o2, 'rubber'), Size(o3, 'big'), Material(o3, 'rubber'), Shape(o3, 'sphere'), LeftOf(o1, o0), FrontOf(o1, o2), FrontOf(o2, o3)
"""Question: A ball is the same material as a matte sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
PRED0=SPHERE(object=OBJ0,var='o0')
PRED1=RUBBER(object=OBJ0,var='o1')
PRED2=SPHERE(object=OBJ0,var='o1')
PRED3=EQUALMATERIAL(object1=OBJ0,var1='o0',object2=OBJ0,var2='o1')
EVENT0=EVENT(predicates=[PRED0,PRED1,PRED2,PRED3])
ANSWER0=EVAL(expr="'yes' if len({EVENT0}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""" # Shape(o0, 'sphere'), Material(o1, 'rubber'), Shape(o1, 'sphere'), EqualMaterial(o0, o1)
]

def create_prompt(inputs,num_prompts=8,method='random',seed=42,group=0,prompt_modules=None):
    if method=='all':
        prompt_examples = CLEVR_CURATED_EXAMPLES
    elif method=='random':
        random.seed(seed)
        prompt_examples = random.sample(CLEVR_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    if prompt_modules is not None:
        prompt_examples = f'{prompt_modules}\n\n{prompt_examples}'
    prompt_examples = f'Think step by step to answer the question.\n\n{prompt_examples}'


    return prompt_examples + "\nQuestion: {question}\nProgram:".format(**inputs)