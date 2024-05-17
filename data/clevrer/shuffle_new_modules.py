import json
import random

if __name__ == '__main__':
    # random seed
    random.seed(0)

    with open("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/3_new_udfs_labels.json", "r") as f:
        data = json.load(f)

    for i in range(len(data['questions'])):
        # randomly shuffle the new modules and save the shuffled data
        random.shuffle(data['questions'][i]['new_modules'])
    with open("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevrer/3_new_udfs_labels_shuffled.json", "w") as f:
        json.dump(data, f, indent=4)