import random

CLEVR_CURATED_EXAMPLES=[
"""Question: a big object o1 is right of the brown cylinder o2 and left of the large brown sphere o3
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=BROWN(object=OBJ0)
OBJ3=CYLINDER(object=OBJ2)
OBJ4=BIG(object=OBJ0)
OBJ5=BROWN(object=OBJ4)
OBJ6=SPHERE(object=OBJ5)
REL1=RIGHTOF(object1=OBJ1,object2=OBJ3)
REL2=LEFTOF(object1=OBJ1,object2=OBJ6)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({OBJ6}) and len({REL1}) and len({REL2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: a brown shiny sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BROWN(object=OBJ0)
OBJ2=METAL(object=OBJ1)
OBJ3=SPHERE(object=OBJ2)
ANSWER0=EVAL(expr="'yes' if len({OBJ3}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: a big blue block
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=BLUE(object=OBJ1)
OBJ3=CUBE(object=OBJ2)
ANSWER0=EVAL(expr="'yes' if len({OBJ3}) else 'no'")
""",
"""Question: a red object right of the big brown object
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=RED(object=OBJ0)
OBJ2=BIG(object=OBJ0)
OBJ3=BROWN(object=OBJ2)
REL1=RIGHTOF(object1=OBJ1,object2=OBJ3)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: a small matte sphere and a matte cylinder
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=SMALL(object=OBJ0)
OBJ2=RUBBER(object=OBJ1)
OBJ3=SPHERE(object=OBJ2)
OBJ4=RUBBER(object=OBJ0)
OBJ5=CYLINDER(object=OBJ4)
ANSWER0=EVAL(expr="'yes' if len({OBJ3}) and len({OBJ5}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
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
""",
"""Question: a brown metallic sphere is in front of the tiny red rubber cylinder
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BROWN(object=OBJ0)
OBJ2=METAL(object=OBJ1)
OBJ3=SPHERE(object=OBJ2)
OBJ4=SMALL(object=OBJ0)
OBJ5=RED(object=OBJ4)
OBJ6=RUBBER(object=OBJ5)
OBJ7=CYLINDER(object=OBJ6)
REL1=FRONTOF(object1=OBJ3,object2=OBJ7)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ7}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: The red cube and the large green cylinder are made of the same material.
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=RED(object=OBJ0)
OBJ2=CUBE(object=OBJ1)
OBJ3=LARGE(object=OBJ0)
OBJ4=GREEN(object=OBJ3)
OBJ5=CYLINDER(object=OBJ4)
REL1=EQUALMATERIAL(object1=OBJ2,object2=OBJ5)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ5}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: A cylinder is in front of the small rubber cylinder and on the left side of the tiny blue metal object.
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=CYLINDER(object=OBJ0)
OBJ2=SMALL(object=OBJ0)
OBJ3=RUBBER(object=OBJ2)
OBJ4=CYLINDER(object=OBJ3)
REL1=FRONTOF(object1=OBJ1,object2=OBJ4)
OBJ5=SMALL(object=OBJ0)
OBJ6=BLUE(object=OBJ5)
OBJ7=METAL(object=OBJ6)
REL2=LEFTOF(object1=OBJ1,object2=OBJ7)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ4}) and len({OBJ7}) and len({REL1}) and len({REL2}) else 'no'")
""",
"""Question: There is a small matte cylinder, which is the same color as the small rubber object in front of the cyan cylinder.
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=SMALL(object=OBJ0)
OBJ2=RUBBER(object=OBJ1)
OBJ3=CYLINDER(object=OBJ2)
OBJ4=SMALL(object=OBJ0)
OBJ5=MATTE(object=OBJ4)
REL1=EQUALCOLOR(object1=OBJ3,object2=OBJ5)
OBJ6=CYAN(object=OBJ0)
OBJ7=CYLINDER(object=OBJ6)
REL2=FRONTOF(object1=OBJ5,object2=OBJ7)
ANSWER0=EVAL(expr="'yes' if len({OBJ3}) and len({OBJ5}) and len({OBJ7}) and len({REL1}) and len({REL2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: A gray cylinder is the same size as a brown matte cylinder
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=GRAY(object=OBJ0)
OBJ2=CYLINDER(object=OBJ1)
OBJ3=BROWN(object=OBJ0)
OBJ4=RUBBER(object=OBJ3)
OBJ5=CYLINDER(object=OBJ4)
REL1=EQUALSIZE(object1=OBJ2,object2=OBJ5)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ5}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: A yellow shiny thing is the same size as a red cylinder
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=YELLOW(object=OBJ0)
OBJ2=METAL(object=OBJ1)
OBJ3=RED(object=OBJ0)
OBJ4=CYLINDER(object=OBJ3)
REL1=EQUALSIZE(object1=OBJ2,object2=OBJ4)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ4}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: A tiny brown thing is on the left side of the thing that is in front of the tiny brown ball behind the small gray shiny thing
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=SMALL(object=OBJ0)
OBJ2=BROWN(object=OBJ1)
OBJ3=SMALL(object=OBJ0)
OBJ4=BROWN(object=OBJ3)
OBJ5=SPHERE(object=OBJ4)
OBJ6=SMALL(object=OBJ0)
OBJ7=GRAY(object=OBJ6)
OBJ8=METAL(object=OBJ7)
REL1=LEFTOF(object1=OBJ2,object2=OBJ0)
REL2=FRONTOF(object1=OBJ0,object2=OBJ5)
REL3=BEHIND(object1=OBJ5,object2=OBJ8)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ5}) and len({OBJ8}) and len({REL1}) and len({REL2}) and len({REL3}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: A tiny metal object is in front of a tiny cylinder on the right side of a tiny brown matte ball
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=SMALL(object=OBJ0)
OBJ2=METAL(object=OBJ1)
OBJ3=SMALL(object=OBJ0)
OBJ4=CYLINDER(object=OBJ3)
OBJ5=SMALL(object=OBJ0)
OBJ6=BROWN(object=OBJ5)
OBJ7=RUBBER(object=OBJ6)
OBJ8=SPHERE(object=OBJ7)
REL1=FRONTOF(object1=OBJ2,object2=OBJ4)
REL2=RIGHTOF(object1=OBJ4,object2=OBJ8)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ4}) and len({OBJ8}) and len({REL1}) and len({REL2}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: A cube is made of the same material as a gray cylinder
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=CUBE(object=OBJ0)
OBJ2=GRAY(object=OBJ0)
OBJ3=CYLINDER(object=OBJ2)
REL1=EQUALMATERIAL(object1=OBJ1,object2=OBJ3)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: There is a object that is in front of a small brown thing
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=SMALL(object=OBJ0)
OBJ2=BROWN(object=OBJ1)
REL1=FRONTOF(object1=OBJ0,object2=OBJ2)
ANSWER0=EVAL(expr="'yes' if len({OBJ0}) and len({OBJ2}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: There is an object that is the same shape as the cyan object
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=CYAN(object=OBJ0)
REL1=EQUALSHAPE(object1=OBJ0,object2=OBJ1)
ANSWER0=EVAL(expr="'yes' if len({OBJ0}) and len({OBJ1}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: A big thing is made of the same material as the gray block.
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=BIG(object=OBJ0)
OBJ2=GRAY(object=OBJ0)
OBJ3=CUBE(object=OBJ2)
REL1=EQUALMATERIAL(object1=OBJ1,object2=OBJ3)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: A tiny gray thing is to the right of the gray sphere that is in front of the metallic object in front of the large metal sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=SMALL(object=OBJ0)
OBJ2=GRAY(object=OBJ1)
OBJ3=GRAY(object=OBJ0)
OBJ4=SPHERE(object=OBJ3)
OBJ5=METAL(object=OBJ0)
OBJ6=LARGE(object=OBJ0)
OBJ7=METAL(object=OBJ6)
OBJ8=SPHERE(object=OBJ7)
REL1=RIGHTOF(object1=OBJ2,object2=OBJ4)
REL2=FRONTOF(object1=OBJ4,object2=OBJ5)
REL3=FRONTOF(object1=OBJ5,object2=OBJ8)
ANSWER0=EVAL(expr="'yes' if len({OBJ2}) and len({OBJ4}) and len({OBJ5}) and len({OBJ8}) and len({REL1}) and len({REL2}) and len({REL3}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: A ball is the same color as a metallic sphere
Program:
OBJ0=LOC(image=IMAGE,object='object')
OBJ1=SPHERE(object=OBJ0)
OBJ2=METAL(object=OBJ0)
OBJ3=SPHERE(object=OBJ2)
REL1=EQUALCOLOR(object1=OBJ1,object2=OBJ3)
ANSWER0=EVAL(expr="'yes' if len({OBJ1}) and len({OBJ3}) and len({REL1}) else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
"""
]

def create_prompt(inputs,num_prompts=8,method='random',seed=42,group=0):
    if method=='all':
        prompt_examples = CLEVR_CURATED_EXAMPLES
    elif method=='random':
        random.seed(seed)
        prompt_examples = random.sample(CLEVR_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f'Think step by step to answer the question.\n\n{prompt_examples}'


    return prompt_examples + "\nQuestion: {question}\nProgram:".format(**inputs)