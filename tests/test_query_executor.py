import yaml
import random
import json
from vocaludf.utils import parse_signature
import logging
import numpy as np
from vocaludf.query_executor import QueryExecutor

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    run_id = 0
    query_id = 0
    random.seed(run_id)
    np.random.seed(run_id)
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    dataset = "clevrer"
    program_with_pixels = True
    num_workers = 8
    parsed_program = {'query': [{'scene_graph': [{'predicate': 'left_of', 'variables': ['o2', 'o0']}, {'predicate': 'color_red', 'variables': ['o0']}, {'predicate': 'front_of', 'variables': ['o1', 'o2']}], 'duration_constraint': 15}, {'scene_graph': [{'predicate': 'far_from', 'variables': ['o1', 'o2']}, {'predicate': 'left_of', 'variables': ['o0', 'o2']}], 'duration_constraint': 5}, {'scene_graph': [{'predicate': 'behind_of', 'variables': ['o2', 'o0']}, {'predicate': 'material_metal', 'variables': ['o1']}], 'duration_constraint': 1}]}
    input_query_file = config[dataset]["input_query_file"]
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    positive_videos = input_query["positive_videos"]
    y_true = [1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"])]
    object_domain = ['object']
    relationship_domain = ['left_of', 'front_of']
    attribute_domain = ['location_left', 'location_top', 'color_gray', 'color_red', 'color_blue', 'color_green', 'shape_cube', 'shape_sphere', 'material_rubber', 'shape_cylinder']
    registered_functions = json.load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r")
    )["test"]
    # registered_functions.append(
    #     {
    #         "signature": "shape_cylinder(o0)",
    #         "description": "whether o0 is a cylinder",
    #         "semantic_interpretation": "",
    #         "function_implementation": "def shape_cylinder(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, height, width):\n  return 'shape_cylinder' in o1_anames",
    #     }
    # )
    available_udf_names = [parse_signature(func["signature"])[0] for func in registered_functions]
    registered_functions.extend([
        {
            "signature": "material_metal(o0)",
            "description": "whether o0 is made of metal",
            "semantic_interpretation": "model",
            "function_implementation": "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/model_udf/clevrer/gpt4v_three_clip/material_metal/lightning_logs/version_18092598/checkpoints/udf-material_metal_run-0_ntrain-100.ckpt",
        },
        {
            "signature": "far_from(o0, o1)",
            "description": "whether o0 is far from o1",
            "semantic_interpretation": "",
            "function_implementation": "def py_far_from(img, o0_oname, o0_x1, o0_y1, o0_x2, o0_y2, o0_anames, o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o0_o1_rnames, o1_o0_rnames, height, width):\n    kwargs = {\"distance_threshold\": 50}\n    threshold = kwargs.get('distance_threshold', 0.1 * (height + width) / 2)\n    centroid_o0 = ((o0_x1 + o0_x2) / 2, (o0_y1 + o0_y2) / 2)\n    centroid_o1 = ((o1_x1 + o1_x2) / 2, (o1_y1 + o1_y2) / 2)\n    distance = ((centroid_o0[0] - centroid_o1[0]) ** 2 + (centroid_o0[1] - centroid_o1[1]) ** 2) ** 0.5\n    return distance > threshold",
        },
        {
            "signature": "behind_of(o0, o1)",
            "description": "whether o0 is behind o1",
            "semantic_interpretation": "model",
            "function_implementation": "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/model_udf/clevrer/gpt4v_three_clip/behind_of/lightning_logs/version_18092598/checkpoints/udf-behind_of_run-0_ntrain-100.ckpt",
        }
    ])
    # materialized_df_names = ["shape_cylinder"]
    materialized_df_names = ['material_metal', 'behind_of']
    on_the_fly_udf_names = ['far_from']
    qe = QueryExecutor(config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_df_names, on_the_fly_udf_names, program_with_pixels, num_workers)
    qe.run(parsed_program, y_true, debug=False)