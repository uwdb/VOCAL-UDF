# read in pkl file
import pickle
import numpy as np
import json
import pprint

# Convert numpy arrays to lists (if necessary)
def convert_np_arrays_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np_arrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_arrays_to_lists(elem) for elem in obj]
    return obj

def convert_pkl_to_json(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    converted_data = convert_np_arrays_to_lists(data)

    # Write the data to a .json file
    with open(file_path.replace('.pkl', '.json'), 'w') as file:
        json.dump(converted_data, file, indent=4)

    # with open(file_path.replace('.pkl', '.txt'), "a") as f:
    #      pprint.pprint(data, stream=f)

if __name__ == '__main__':
    convert_pkl_to_json("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/charades/person_bbox.pkl")
    convert_pkl_to_json("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/charades/object_bbox_and_relationship.pkl")