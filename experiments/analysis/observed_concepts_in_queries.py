from src.utils import program_to_dsl, dsl_to_program
import json
import yaml
import os
config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

def eval_observed_concepts_in_queries(dataset, query_class_names):
    observed_concepts = set()
    for query_class_name in query_class_names:
        with open(os.path.join(config["data_dir"], dataset, f"{query_class_name}.json"), "r") as f:
            data = json.load(f)
        for query in data["questions"]:
            program = dsl_to_program(query["dsl"])
            for sg in program:
                for pred in sg["scene_graph"]:
                    observed_concepts.add(pred["predicate"])
    print(f"Observed concepts in queries: {observed_concepts}, {len(observed_concepts)}")

if __name__ == "__main__":
    # Clevrer
    print("Clevrer")
    dataset = "clevrer"
    query_class_names = [
        "3_new_udfs_labels",
    ]
    eval_observed_concepts_in_queries(dataset, query_class_names)

    # CityFlow
    print("CityFlow")
    dataset = "cityflow"
    query_class_names = [
        "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737",
        "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737"
    ]
    eval_observed_concepts_in_queries(dataset, query_class_names)

    # Charades
    print("Charades")
    dataset = "charades"
    query_class_names = [
        "unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2",
        "unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2",
        "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2"
    ]
    eval_observed_concepts_in_queries(dataset, query_class_names)