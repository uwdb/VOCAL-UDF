import random

CLEVR_CURATED_EXAMPLES=[
"""Question: a big object o1 is right of the gray cube o2 and left of the large gray sphere o3
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=GRAY(object=OBJ0)
OBJ3=CUBE(object=OBJ2)
OBJ4=BIG(object=OBJ0)
OBJ5=GRAY(object=OBJ4)
OBJ6=SPHERE(object=OBJ5)
REL1=LEFTOF(object1=OBJ3,object2=OBJ1)
REL2=LEFTOF(object1=OBJ1,object2=OBJ6)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({OBJ6}) and len({REL1}) and len({REL2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Color(o1, 'gray'), Shape(o1, 'cube'), Size(o2, 'big'), Color(o2, 'gray'), Shape(o2, 'sphere'), LeftOf(o1, o0), LeftOf(o0, o2)
"""Question: a gray rubber sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=GRAY(object=OBJ0)
OBJ2=RUBBER(object=OBJ1)
OBJ3=SPHERE(object=OBJ2)
ANSWER0=EVAL(expr="'yes' if len({OBJ3}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'gray'), Material(o0, 'rubber'), Shape(o0, 'sphere')
"""Question: a big blue block
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=BLUE(object=OBJ1)
OBJ3=CUBE(object=OBJ2)
ANSWER0=EVAL(expr="'yes' if len({OBJ3}) else 'no'")
""", # Size(o0, 'big'), Color(o0, 'blue'), Shape(o0, 'cube')
"""Question: a red object right of the big gray object
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=RED(object=OBJ0)
OBJ2=BIG(object=OBJ0)
OBJ3=GRAY(object=OBJ2)
REL1=LEFTOF(object1=OBJ3,object2=OBJ1)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'red'), Size(o1, 'big'), Color(o1, 'gray'), LeftOf(o1, o0)
"""Question: a big matte sphere and a matte sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=RUBBER(object=OBJ1)
OBJ3=SPHERE(object=OBJ2)
OBJ4=RUBBER(object=OBJ0)
OBJ5=SPHERE(object=OBJ4)
ANSWER0=EVAL(expr="'yes' if len({OBJ3}) and len({OBJ5}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Material(o0, 'rubber'), Shape(o0, 'sphere'), Material(o1, 'rubber'), Shape(o1, 'sphere')
"""Question: a matte thing is in front of the gray ball, and it is the same size as the gray object.
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=RUBBER(object=OBJ0)
OBJ2=SPHERE(object=OBJ0)
OBJ3=GRAY(object=OBJ2)
REL1=FRONTOF(object1=OBJ1,object2=OBJ3)
REL2=EQUALSIZE(object1=OBJ1,object2=OBJ3)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({REL1}) and len({REL2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Material(o0, 'rubber'), Shape(o1, 'sphere'), Color(o1, 'gray'), FrontOf(o0, o1), EqualSize(o0, o1)
"""Question: a gray rubber sphere is in front of the huge red rubber cube
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=GRAY(object=OBJ0)
OBJ2=RUBBER(object=OBJ1)
OBJ3=SPHERE(object=OBJ2)
OBJ4=BIG(object=OBJ0)
OBJ5=RED(object=OBJ4)
OBJ6=RUBBER(object=OBJ5)
OBJ7=CUBE(object=OBJ6)
REL1=FRONTOF(object1=OBJ3,object2=OBJ7)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ7}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'gray'), Material(o0, 'rubber'), Shape(o0, 'sphere'), Size(o1, 'big'), Color(o1, 'red'), Material(o1, 'rubber'), Shape(o1, 'cube'), FrontOf(o0, o1)
"""Question: The red cube and the large green sphere are made of the same material.
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=RED(object=OBJ0)
OBJ2=CUBE(object=OBJ1)
OBJ3=BIG(object=OBJ0)
OBJ4=GREEN(object=OBJ3)
OBJ5=SPHERE(object=OBJ4)
REL1=EQUALMATERIAL(object1=OBJ2,object2=OBJ5)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ5}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'red'), Shape(o0, 'cube'), Size(o1, 'big'), Color(o1, 'green'), Shape(o1, 'sphere'), EqualMaterial(o0, o1)
"""Question: A cube is in front of the big rubber sphere and on the left side of the big blue rubber object.
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=CUBE(object=OBJ0)
OBJ2=BIG(object=OBJ0)
OBJ3=RUBBER(object=OBJ2)
OBJ4=SPHERE(object=OBJ3)
REL1=FRONTOF(object1=OBJ1,object2=OBJ4)
OBJ5=BIG(object=OBJ0)
OBJ6=BLUE(object=OBJ5)
OBJ7=RUBBER(object=OBJ6)
REL2=LEFTOF(object1=OBJ1,object2=OBJ7)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ4}) and len({OBJ7}) and len({REL1}) and len({REL2}) else 'no'")
""", # Shape(o0, 'cube'), FrontOf(o0, o1), Size(o1, 'big'), Material(o1, 'rubber'), Shape(o1, 'sphere'), Size(o2, 'big'), Color(o2, 'blue'), Material(o2, 'rubber'), LeftOf(o0, o2)
"""Question: There is a big matte sphere, which is the same material as the big rubber object in front of the blue cube.
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=RUBBER(object=OBJ1)
OBJ3=SPHERE(object=OBJ2)
OBJ4=BIG(object=OBJ0)
OBJ5=RUBBER(object=OBJ4)
REL1=EQUALMATERIAL(object1=OBJ3,object2=OBJ5)
OBJ6=BLUE(object=OBJ0)
OBJ7=CUBE(object=OBJ6)
REL2=FRONTOF(object1=OBJ5,object2=OBJ7)
ANSWER0=EVAL(expr="'yes' if len({OBJ3}) and len({OBJ5}) and len({OBJ7}) and len({REL1}) and len({REL2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Material(o0, 'rubber'), Shape(o0, 'sphere'), Size(o1, 'big'), Material(o1, 'rubber'), EqualMaterial(o0, o1), Color(o2, 'blue'), Shape(o2, 'cube'), FrontOf(o1, o2)
"""Question: A gray sphere is the same size as a gray matte sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=GRAY(object=OBJ0)
OBJ2=SPHERE(object=OBJ1)
OBJ3=GRAY(object=OBJ0)
OBJ4=RUBBER(object=OBJ3)
OBJ5=SPHERE(object=OBJ4)
REL1=EQUALSIZE(object1=OBJ2,object2=OBJ5)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ5}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'gray'), Shape(o0, 'sphere'), Color(o1, 'gray'), Material(o1, 'rubber'), Shape(o1, 'sphere'), EqualSize(o0, o1)
"""Question: A green rubber thing is the same size as a red cube
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=GREEN(object=OBJ0)
OBJ2=RUBBER(object=OBJ1)
OBJ3=RED(object=OBJ0)
OBJ4=CUBE(object=OBJ3)
REL1=EQUALSIZE(object1=OBJ2,object2=OBJ4)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ4}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'green'), Material(o0, 'rubber'), Color(o1, 'red'), Shape(o1, 'cube'), EqualSize(o0, o1)
"""Question: A large gray thing is on the left side of the thing that is in front of the large gray ball behind the large gray matte thing
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=GRAY(object=OBJ1)
OBJ3=BIG(object=OBJ0)
OBJ4=GRAY(object=OBJ3)
OBJ5=SPHERE(object=OBJ4)
OBJ6=BIG(object=OBJ0)
OBJ7=GRAY(object=OBJ6)
OBJ8=RUBBER(object=OBJ7)
REL1=LEFTOF(object1=OBJ2,object2=OBJ0)
REL2=FRONTOF(object1=OBJ0,object2=OBJ5)
REL3=FRONTOF(object1=OBJ8,object2=OBJ5)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ5}) and len({OBJ8}) and len({REL1}) and len({REL2}) and len({REL3}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Color(o0, 'gray'), Size(o1, 'big'), Color(o1, 'gray'), Shape(o1, 'sphere'), Size(o2, 'big'), Color(o2, 'gray'), Material(o2, 'rubber'), LeftOf(o0, o3), FrontOf(o3, o1), FrontOf(o2, o1)
"""Question: A large matte object is in front of a large cube on the right side of a large gray matte ball
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=RUBBER(object=OBJ1)
OBJ3=BIG(object=OBJ0)
OBJ4=CUBE(object=OBJ3)
OBJ5=BIG(object=OBJ0)
OBJ6=GRAY(object=OBJ5)
OBJ7=RUBBER(object=OBJ6)
OBJ8=SPHERE(object=OBJ7)
REL1=FRONTOF(object1=OBJ2,object2=OBJ4)
REL2=LEFTOF(object1=OBJ8,object2=OBJ4)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ4}) and len({OBJ8}) and len({REL1}) and len({REL2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Material(o0, 'rubber'), Size(o1, 'big'), Shape(o1, 'cube'), Size(o2, 'big'), Color(o2, 'gray'), Material(o2, 'rubber'), Shape(o2, 'sphere'), FrontOf(o0, o1), LeftOf(o2, o1)
"""Question: A cube is made of the same material as a gray sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=CUBE(object=OBJ0)
OBJ2=GRAY(object=OBJ0)
OBJ3=SPHERE(object=OBJ2)
REL1=EQUALMATERIAL(object1=OBJ1,object2=OBJ3)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Shape(o0, 'cube'), Color(o1, 'gray'), Shape(o1, 'sphere'), EqualMaterial(o0, o1)
"""Question: There is a object that is in front of a big green thing
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=GREEN(object=OBJ1)
REL1=FRONTOF(object1=OBJ0,object2=OBJ2)
ANSWER0=EVAL(expr="'yes' if len({OBJ0}) and len({OBJ2}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Color(o0, 'green'), FrontOf(o1, o0)
"""Question: There is an object that is the same size as the blue object
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BLUE(object=OBJ0)
REL1=EQUALSIZE(object1=OBJ0,object2=OBJ1)
ANSWER0=EVAL(expr="'yes' if len({OBJ0}) and len({OBJ1}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Color(o0, 'blue'), EqualSize(o0, o1)
"""Question: A big thing is made of the same material as the gray block.
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=GRAY(object=OBJ0)
OBJ3=CUBE(object=OBJ2)
REL1=EQUALMATERIAL(object1=OBJ1,object2=OBJ3)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Color(o1, 'gray'), Shape(o1, 'cube'), EqualMaterial(o0, o1)
"""Question: A large gray thing is to the right of the gray sphere that is in front of the rubber object in front of the large rubber sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=GRAY(object=OBJ1)
OBJ3=GRAY(object=OBJ0)
OBJ4=SPHERE(object=OBJ3)
OBJ5=RUBBER(object=OBJ0)
OBJ6=BIG(object=OBJ0)
OBJ7=RUBBER(object=OBJ6)
OBJ8=SPHERE(object=OBJ7)
REL1=LEFTOF(object1=OBJ4,object2=OBJ2)
REL2=FRONTOF(object1=OBJ4,object2=OBJ5)
REL3=FRONTOF(object1=OBJ5,object2=OBJ8)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ4}) and len({OBJ5}) and len({OBJ8}) and len({REL1}) and len({REL2}) and len({REL3}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""", # Size(o0, 'big'), Color(o0, 'gray'), Color(o1, 'gray'), Shape(o1, 'sphere'), Material(o2, 'rubber'), Size(o3, 'big'), Material(o3, 'rubber'), Shape(o3, 'sphere'), LeftOf(o1, o0), FrontOf(o1, o2), FrontOf(o2, o3)
"""Question: A ball is the same material as a matte sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=SPHERE(object=OBJ0)
OBJ2=RUBBER(object=OBJ0)
OBJ3=SPHERE(object=OBJ2)
REL1=EQUALMATERIAL(object1=OBJ1,object2=OBJ3)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({REL1}) else 'no'")
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