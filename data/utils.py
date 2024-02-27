# Read a json file, randomly select 100 elements from the "questions" key and write them to a new json file
import json
import random
import os
from src.utils import program_to_dsl, dsl_to_program

def create_sample_json():
    with open('/home/enhao/visprog/datasets/clevr/questions/CLEVR_train_questions.json') as f:
        data = json.load(f)

    random.seed(42)
    data['questions'] = random.sample(data['questions'],100)

    with open('/home/enhao/visprog/datasets/clevr/questions/CLEVR_train_questions_100.json','w') as f:
        json.dump(data,f)

def key_value_attr_to_aname(dir_name):
    # Convert attribute predicates from key-value format to attribute name format
    # e.g., Color(o1, 'red') -> Color_red(o1)
    def rewrite_dsl(dsl):
        program = dsl_to_program(dsl)
        for sg in program:
            for pred in sg['scene_graph']:
                if pred['parameter']:
                    if pred['predicate'] in ['Far', 'Near']:
                        pred['predicate'] = pred['predicate']
                    else:
                        pred['predicate'] = pred['predicate'] + '_' + pred['parameter']
                    pred['parameter'] = None
        return program_to_dsl(program, rewrite_variables=False, sort_variables=False)

    for file in os.listdir(dir_name):
        if file.endswith('_labels.json'):
            if file == 'vocab_clevrer.json':
                continue
            with open(os.path.join(dir_name, file)) as f:
                data = json.load(f)
            for q in data['questions']:
                q["dsl"] = rewrite_dsl(q["dsl"])
            with open(os.path.join(dir_name, file), 'w') as f:
                json.dump(data, f, indent=4)

if __name__ == "__main__":
    key_value_attr_to_aname("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr")