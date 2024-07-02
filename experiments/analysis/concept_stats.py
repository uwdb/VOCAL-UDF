import json
import yaml
import os
import duckdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from vocaludf.utils import (
    replace_slot,
    parse_signature,
    transform_function,
    PredImageDataset
)
from openai import OpenAI
import re

client = OpenAI()

def init_table(conn, dataset):
    # TODO: add object UDFs
    sql = f"""
        CREATE TEMPORARY TABLE one_object AS
        SELECT
            o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
            o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
            m.height AS height, m.width AS width
        FROM {dataset}_objects o
        LEFT OUTER JOIN {dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
        JOIN {dataset}_metadata m ON o.vid = m.vid AND o.fid = m.fid
        GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2, m.height, m.width
    """
    print(f"Create one_object table:\n{sql}")
    one_object_df = conn.execute(sql).df()

    sql = f"""
        SELECT
            o1.vid AS vid, o1.fid AS fid,
            o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
            o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2,
            COALESCE(ARRAY_AGG(r.rname), ARRAY[]::varchar[]) AS o1_o2_gt_rnames,
            m.height AS height, m.width AS width
        FROM {dataset}_objects o1
        JOIN {dataset}_objects o2 ON o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid <> o2.oid
        JOIN {dataset}_metadata m ON o1.vid = m.vid AND o1.fid = m.fid
        LEFT OUTER JOIN {dataset}_relationships r ON o1.vid = r.vid AND o1.fid = r.fid AND o1.oid = r.oid1 AND o2.oid = r.oid2
        GROUP BY o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2, o2.oid, o2.oname, o2.x1, o2.y1, o2.x2, o2.y2, m.height, m.width
    """
    print(f"Create two_objects table:\n{sql}")
    two_objects_df = conn.execute(sql).df()
    return one_object_df, two_objects_df


def llm_filter_relevant_objects(udf_signature, udf_description, prompt_config, object_domain, run_id):
    filter_objects_prompt = replace_slot(
        prompt_config["filter_training_data_subject_target"],
        {
            "object_classes": object_domain,
            "udf_signature": udf_signature,
            "udf_description": udf_description,
        },
    )
    print("filter_objects_prompt: {}".format(filter_objects_prompt))
    for trial in range(3):  # Retry 3 times
        response = None
        try:
            print(f"trial: {trial}")
            response = client.chat.completions.create(
                model="gpt-4-turbo-2024-04-09",
                messages=[
                    {"role": "user", "content": filter_objects_prompt},
                ],
                temperature=config["udf_generator"]["temperature"],
                top_p=config["udf_generator"]["top_p"],
                seed=run_id * 42 + trial,
            )
            answer = eval(
                "\n\n".join(
                    re.findall(
                        r"```json\n(.*?)```",
                        response.choices[0].message.content,
                        re.DOTALL,
                    )
                )
            )
            filtered_subjects = answer["subjects"]
            filtered_targets = answer["targets"]
            return filtered_subjects, filtered_targets
        except Exception as e:
            print("ERROR: failed to filter relevant objects: {}".format(e))
            print(response)


if __name__ == "__main__":
    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))
    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )
    dataset = "gqa"
    conn = duckdb.connect(
        database=os.path.join(config["db_dir"], "annotations.duckdb"),
        read_only=True,
    )

    one_object_df, two_objects_df = init_table(conn, dataset)

    test_inputs = [
        ["wearing(o0, o1)", "Whether o0 is wearing o1.", "wearing"],
        ["holding(o0, o1)", "Whether o0 is holding o1", "holding"],
        ["walking_on(o0, o1)", "Whether o0 is walking on o1.", "walking_on"],
        ["carrying(o0, o1)", "Whether o0 is carrying o1.", "carrying"],
    ]

    object_k = 50
    relationship_k = 30
    # Top-50 objects: ['window', 'man', 'shirt', 'tree', 'wall', 'person', 'building', 'ground', 'sky', 'sign', 'head', 'pole', 'hand', 'grass', 'hair', 'leg', 'car', 'woman', 'leaves', 'trees', 'table', 'ear', 'pants', 'people', 'eye', 'water', 'door', 'fence', 'nose', 'wheel', 'chair', 'floor', 'arm', 'jacket', 'hat', 'shoe', 'tail', 'clouds', 'leaf', 'face', 'letter', 'plate', 'number', 'windows', 'shorts', 'road', 'flower', 'sidewalk', 'bag', 'helmet']

    # Stats before filtering top-k objects
    print("stats before filtering top-k objects")
    top_k_relationships = conn.execute(f"""
        SELECT rname, COUNT(*) * 100 / {len(two_objects_df)}
        FROM {dataset}_relationships
        GROUP BY rname
        ORDER BY COUNT(*) DESC
        LIMIT {relationship_k}
    """).df()
    print("top_k_relationships", top_k_relationships.to_string(index=False))


    # Stats after filtering top-k objects
    print("stats after filtering top-k objects")
    top_k_objects = conn.execute(f"""
        SELECT oname, COUNT(*) FROM {dataset}_objects GROUP BY oname ORDER BY COUNT(*) DESC LIMIT {object_k}
    """).df()["oname"].tolist()
    print("top_k_objects", top_k_objects)

    obj_parameters = ','.join('?' for _ in top_k_objects)
    filtered_relationship_count = conn.execute(f"""
        SELECT COUNT(*) as count
        FROM two_objects_df
        WHERE o1_oname = ANY([{obj_parameters}]) AND o2_oname = ANY([{obj_parameters}])
    """, top_k_objects + top_k_objects).df()["count"][0]

    top_k_relationships = conn.execute(f"""
        SELECT rname, COUNT(*) * 100 / {filtered_relationship_count}
        FROM {dataset}_relationships r, {dataset}_objects o1, {dataset}_objects o2
        WHERE r.vid = o1.vid AND r.fid = o1.fid AND r.oid1 = o1.oid AND r.vid = o2.vid AND r.fid = o2.fid AND r.oid2 = o2.oid
        AND o1.oname = ANY([{obj_parameters}]) AND o2.oname = ANY([{obj_parameters}])
        GROUP BY rname
        ORDER BY COUNT(*) DESC
        LIMIT {relationship_k}
    """, top_k_objects + top_k_objects).df()
    print("top_k_relationships", top_k_relationships.to_string(index=False))


    # Stats after filtering top-k objects and human-object relationships
    for test_input in test_inputs:
        udf_signature, udf_description, udf_name = test_input
        print(f"udf_signature: {udf_signature}, udf_description: {udf_description}, udf_name: {udf_name}")
        filtered_subjects, filtered_targets = llm_filter_relevant_objects(udf_signature=udf_signature, udf_description=udf_description, prompt_config=prompt_config, object_domain=top_k_objects, run_id=0)
        print("filtered_subjects", filtered_subjects)
        print("filtered_targets", filtered_targets)

        filtered_subject_parameters = ','.join('?' for _ in filtered_subjects)
        filtered_target_parameters = ','.join('?' for _ in filtered_targets)
        filtered_relationship_count = conn.execute(f"""
            SELECT COUNT(*) as count
            FROM (
                SELECT vid, fid, o1_oid, o2_oid
                FROM two_objects_df
                WHERE o1_oname = ANY([{filtered_subject_parameters}]) AND o2_oname = ANY([{filtered_target_parameters}])
                GROUP BY vid, fid, o1_oid, o2_oid
            )
        """, filtered_subjects + filtered_targets).df()["count"][0]

        top_k_relationships = conn.execute(f"""
            SELECT rname, COUNT(*) * 100 / {filtered_relationship_count}
            FROM {dataset}_relationships r, {dataset}_objects o1, {dataset}_objects o2
            WHERE r.vid = o1.vid AND r.fid = o1.fid AND r.oid1 = o1.oid AND r.vid = o2.vid AND r.fid = o2.fid AND r.oid2 = o2.oid
            AND o1.oname = ANY([{filtered_subject_parameters}]) AND o2.oname = ANY([{filtered_target_parameters}])
            GROUP BY rname
            ORDER BY COUNT(*) DESC
            LIMIT {relationship_k}
        """, filtered_subjects + filtered_targets).df()
        print("top_k_relationships", top_k_relationships.to_string(index=False))


# wearing (# / 100): ['man', 'person', 'woman', 'people', 'shirt', 'pants', 'jacket', 'hat', 'helmet', 'shoe', 'shorts']
# holding (# 0.9 / 100): ['man', 'person', 'woman', 'people', 'bag', 'plate', 'letter', 'helmet', 'hat', 'shoe', 'jacket', 'shirt', 'pants', 'shorts', 'sign', 'pole', 'chair', 'table', 'flower']
# walking on (# 11 / 100): ['man', 'woman', 'person', 'people', 'ground', 'floor', 'road', 'sidewalk']
# carrying (# 3 / 100): ['man', 'woman', 'person', 'hand', 'arm', 'bag', 'helmet', 'letter', 'plate']