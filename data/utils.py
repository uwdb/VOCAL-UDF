# Read a json file, randomly select 100 elements from the "questions" key and write them to a new json file
import json
import random

with open('/home/enhao/visprog/datasets/clevr/questions/CLEVR_train_questions.json') as f:
    data = json.load(f)

random.seed(42)
data['questions'] = random.sample(data['questions'],100)

with open('/home/enhao/visprog/datasets/clevr/questions/CLEVR_train_questions_100.json','w') as f:
    json.dump(data,f)
