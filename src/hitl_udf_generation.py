import argparse
from src.utils import dsl_to_program, postgres_execute_cache_sequence
import psycopg2 as psycopg
from psycopg2 import pool
import re
import json
from openai import OpenAI
import logging
import pandas as pd
import numpy as np
import time
from sklearn.metrics import f1_score
import importlib
import copy
import os
import yaml

client = OpenAI()

# Read csv file into pandas dataframe
df = pd.read_csv("/home/enhao/EQUI-VOCAL/postgres/obj_clevrer.csv")
# relation schema: obj_clevrer(oid INT, vid INT, fid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float)
# set column names
df.columns = ['oid', 'vid', 'fid', 'shape', 'color', 'material', 'x1', 'y1', 'x2', 'y2']
df_filtered_train = df[df['vid'] < 200]
df_filtered_test = df[(df['vid'] >= 200) & (df['fid'] == 63)]

def construct_df_object_pairs(df_filtered):
    # execute "SELECT * FROM obj_clevrer o1, obj_clevrer o2 WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid AND vid < 200 ORDER BY o1.vid, o1.fid, o1.oid, o2.oid;" using pandas

    # Set 'vid' and 'fid' as indices
    df_filtered = df_filtered.set_index(['vid', 'fid'])

    # Perform the self-join
    df_object_pairs = df_filtered.join(df_filtered, lsuffix='_o1', rsuffix='_o2')

    # Filter out rows where 'oid_o1' is equal to 'oid_o2'
    df_object_pairs = df_object_pairs[df_object_pairs['oid_o1'] != df_object_pairs['oid_o2']]

    # Sorting the result
    df_object_pairs = df_object_pairs.sort_values(by=['vid', 'fid', 'oid_o1', 'oid_o2'])

    df_object_pairs['o1'] = df_object_pairs.apply(lambda row: {
        'shape': row['shape_o1'],
        'color': row['color_o1'],
        'material': row['material_o1'],
        'x1': row['x1_o1'],
        'y1': row['y1_o1'],
        'x2': row['x2_o1'],
        'y2': row['y2_o1']
    }, axis=1)

    df_object_pairs['o2'] = df_object_pairs.apply(lambda row: {
        'shape': row['shape_o2'],
        'color': row['color_o2'],
        'material': row['material_o2'],
        'x1': row['x1_o2'],
        'y1': row['y1_o2'],
        'x2': row['x2_o2'],
        'y2': row['y2_o2']
    }, axis=1)

    # Reset index if needed
    df_object_pairs = df_object_pairs.reset_index()

    return df_object_pairs

def construct_df_object(df_filtered):
    # execute "SELECT * FROM obj_clevrer o1, obj_clevrer o2 WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid AND vid < 200 ORDER BY o1.vid, o1.fid, o1.oid, o2.oid;" using pandas

    # Set 'vid' and 'fid' as indices
    df_filtered = df_filtered.set_index(['vid', 'fid'])

    # Sorting the result
    df_object = df_filtered.sort_values(by=['vid', 'fid', 'oid'])

    df_object['o1'] = df_object.apply(lambda row: {
        'shape': row['shape'],
        'color': row['color'],
        'material': row['material'],
        'x1': row['x1'],
        'y1': row['y1'],
        'x2': row['x2'],
        'y2': row['y2']
    }, axis=1)

    # Reset index if needed
    df_object = df_object.reset_index()

    return df_object

df_object_pairs_train = construct_df_object_pairs(df_filtered_train)
df_object_pairs_test = construct_df_object_pairs(df_filtered_test)

df_object_train = construct_df_object(df_filtered_train)
df_object_test = construct_df_object(df_filtered_test)

labeled_index = []

def response_text(openai_resp):
    return openai_resp.choices[0].message.content

def replace_slot(text, entries):
    new_text = text
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        new_text = new_text.replace("{{" + key +"}}", value.replace('"', "'"))
    return new_text

def generate_udf(udf_id, prompt, semantic_interpretation):
    logger.info("generate_udf_prompt: {}".format(prompt))
    for retry in range(3):
        try:
            response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                "role": "system",
                "content": "You are an expert programming assistant."
                },
                {
                "role": "user",
                "content": prompt
                }
            ],
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
            function_implementation = "\n\n".join(re.findall(r"```python\n(.*?)```", response_text(response), re.DOTALL))
            kwargs_signature = "\n\n".join(re.findall(r"```json\n(.*?)```", response_text(response), re.DOTALL))
            kwargs_signature = json.loads(kwargs_signature)
            # write to json file
            with open(os.path.join("/home/enhao/EQUI-VOCAL/outputs/udf_generation", udf_class, "{}_{}.json".format(udf_class, udf_id)), "w") as f:
                json.dump({"semantic_interpretation": semantic_interpretation, "function_implementation": function_implementation, "kwargs_signature": kwargs_signature}, f)
            break
        except Exception as e:
            print("ERROR: failed to generate UDF", e)
            print(response)
            print(response_text(response))

def generate_semantic_interpretation(generate_semantic_interpretation_prompt):
    logger.info("generate_semantic_interpretation_prompt: {}".format(generate_semantic_interpretation_prompt))
    try:
        response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
            "role": "system",
            "content": "You are an expert programming assistant."
            },
            {
            "role": "user",
            "content": generate_semantic_interpretation_prompt
            }
        ],
        temperature=0.7,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        text = response_text(response)
        # split into interpretations, which are separated by a line break, andremove the prefix "1. ", "2. ", etc.
        interpretations = [re.sub(r"^\d+\. ", "", interpretation) for interpretation in text.split("\n") if interpretation != ""]
        logger.info("semantic interpretations: {}".format(interpretations))
        return interpretations
    except Exception as e:
        print(e)
        print(response)
        print(response_text(response))

# def compute_udf_score(gt_udf, udf_candidate, new_labeled_index, udf_generation_name, df_object_pairs):
def compute_udf_score(gt_udf, udf_candidate, udf_generation_name, n_obj, df, df_newly_labeled=None):
    """
    Compute the F1 score of the UDF candidate on the data (train or test), using the ground truth UDF as the label
    if df_newly_labeled is provided, also compute the number of misclassified samples of them
    """
    try:
        # For each sampled row in df, construct o1 and o2
        kwargs = {}
        for k, v in udf_candidate["kwargs_signature"].items():
            kwargs[k] = float(v["default"])


        namesapce = {}
        exec(udf_candidate["function_implementation"], namesapce)
        udf_function = namesapce[udf_generation_name]
        if n_obj == 1:
            y_pred = df.apply(lambda row: udf_function(row["o1"], **kwargs), axis=1)
        elif n_obj == 2:
            y_pred = df.apply(lambda row: udf_function(row["o1"], row["o2"], **kwargs), axis=1)
        if df_newly_labeled is not None:
            if n_obj == 1:
                y_pred_new = df_newly_labeled.apply(lambda row: udf_function(row["o1"], **kwargs), axis=1)
            elif n_obj == 2:
                y_pred_new = df_newly_labeled.apply(lambda row: udf_function(row["o1"], row["o2"], **kwargs), axis=1)
    except Exception as e:
        # print("ERROR: failed to execute udf_candidate {}: {}".format(i, e))
        y_pred = df.apply(lambda row: False, axis=1)
        if df_newly_labeled is not None:
            y_pred_new = df_newly_labeled.apply(lambda row: False, axis=1)

    if n_obj == 1:
        y_true = df.apply(lambda row: gt_udf(row["o1"]), axis=1)
    elif n_obj == 2:
        y_true = df.apply(lambda row: gt_udf(row["o1"], row["o2"]), axis=1)
    score = f1_score(y_true, y_pred, zero_division=1.0)

    logger.info("positive: {}, negative: {}".format(sum(y_true), len(y_true) - sum(y_true)))

    if df_newly_labeled is not None:
        if n_obj == 1:
            y_true_new = df_newly_labeled.apply(lambda row: gt_udf(row["o1"]), axis=1)
        elif n_obj == 2:
            y_true_new = df_newly_labeled.apply(lambda row: gt_udf(row["o1"], row["o2"]), axis=1)
        # Count the number of misclassifications for the new samples
        num_misclassified = np.sum(np.array(y_true_new != y_pred_new)*1)
        return score, num_misclassified
    else:
        return score

def _compute_u_t(posterior_t, predictions_c):
    # Initialize possible u_t's
    u_t_list = np.zeros(2)

    # Repeat for each class
    for c in [0, 1]:
        # Compute the loss of models if the label of the streamed data is "c"
        loss_c = np.array(predictions_c != c)*1
        #
        # Compute the respective u_t value (conditioned on class c)
        term1 = np.inner(posterior_t, loss_c)
        u_t_list[c] = term1*(1-term1)

    # Return the final u_t
    u_t = np.max(u_t_list)

    return u_t

def select_sample(udf_candidates_with_scores, udf_generation_name, df_train, n_obj):
    # sample a subset of videos during each iteration
    n_sampled_videos = 500

    prediction_matrix = []
    _start = time.time()

    # query_list = [query_graph.program for query_graph, _ in self.candidate_queries[:self.pool_size]]
    # print("query pool", [program_to_dsl(query, self.rewrite_variables) for query in query_list])
    unlabeled_index = np.setdiff1d(np.arange(total_samples_train), labeled_index, assume_unique=True)

    # If more than n_sampled_videos videos, sample n_sampled_videos videos
    if len(unlabeled_index) > n_sampled_videos:
        sampled_index = np.random.choice(unlabeled_index, n_sampled_videos, replace=False)
    else:
        sampled_index = unlabeled_index

    for udf_id, udf_candidate, _, _ in udf_candidates_with_scores:
        try:
            # For each sampled row in df_train, construct o1 and o2
            kwargs = {}
            for k, v in udf_candidate["kwargs_signature"].items():
                kwargs[k] = float(v["default"])

            namesapce = {}
            exec(udf_candidate["function_implementation"], namesapce)
            udf_function = namesapce[udf_generation_name]
            if n_obj == 1:
                result = df_train.loc[sampled_index].apply(lambda row: udf_function(row["o1"], **kwargs), axis=1)
            elif n_obj == 2:
                result = df_train.loc[sampled_index].apply(lambda row: udf_function(row["o1"], row["o2"], **kwargs), axis=1)
        except Exception as e:
            print("ERROR: failed to execute udf_candidate {}: {}".format(udf_id, e))
            result = df_train.loc[sampled_index].apply(lambda row: False, axis=1)
        prediction_matrix.append(result.values)

    prediction_matrix = np.array(prediction_matrix).transpose() # (n_samples, n_udfs)
    logger.info("constructing prediction matrix took {} seconds".format(time.time()-_start))
    logger.info("prediction_matrix size {}".format(prediction_matrix.shape))

    eta_0 = np.sqrt(np.log(len(udf_candidates_with_scores))/2)

    # Use F1-scores as weights
    posterior_t = [score for _, _, score, _ in udf_candidates_with_scores]
    # Use the original weights as in the paper
    # eta = eta_0 / np.sqrt(n_sampled_videos)
    # loss_t = [loss_t for _, _, _, loss_t in udf_candidates_with_scores]
    # posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))

    posterior_t  /= np.sum(posterior_t)  # normalized weight

    logger.info("query weights {}".format(posterior_t))
    entropy_list = np.zeros(len(sampled_index))
    for i in range(len(sampled_index)):
        entropy_list[i] = _compute_u_t(posterior_t, prediction_matrix[i, :])
    ind = np.argsort(-entropy_list)
    logger.info("entropy list {}".format(entropy_list[ind]))
    # df_object_pairs_train[sampled_index[ind]].apply(lambda row: logger.info("o1: {}, o2: {}".format(row["o1"], row["o2"])), axis=1)
    logger.info("sampled index {}".format(sampled_index[ind]))
    # find argmax of entropy (top k)
    max_entropy_index = sampled_index[np.argmax(entropy_list)]
    return [max_entropy_index]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--udf_class", type=str, help="UDF class name we want to generate")
    parser.add_argument("--n_obj", type=int, help="number of objects in the UDF signature")
    parser.add_argument("--udf_description", type=str, help="UDF description")
    parser.add_argument("--gt_udf_impl", type=str, help="ground truth UDF implementation")
    # parser.add_argument("--udf_generation_name", type=str, help="name of the function that GPT will generate")
    parser.add_argument("--log_name_suffix", type=str, help="log name suffix")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--stage", type=str, help="stage of the experiment (udf_generation, udf_selection, or all)")

    args = parser.parse_args()
    gt_udf_impl = args.gt_udf_impl
    udf_class = args.udf_class
    n_obj= args.n_obj
    udf_description = args.udf_description
    udf_generation_name = "py_{}".format(udf_class)
    log_name_suffix = args.log_name_suffix
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    num_interpretations = args.num_interpretations
    stage = args.stage

    if n_obj == 1:
        df_train = df_object_train
        df_test = df_object_test
    elif n_obj == 2:
        df_train = df_object_pairs_train
        df_test = df_object_pairs_test

    total_samples_train = len(df_train)
    total_samples_test = len(df_test)

    """
    Set up logging
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a directory if it doesn't already exist
    os.makedirs(os.path.join('/home/enhao/EQUI-VOCAL/logs', udf_class), exist_ok=True)
    os.makedirs(os.path.join('/home/enhao/EQUI-VOCAL/outputs/udf_generation', udf_class), exist_ok=True)

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(os.path.join('/home/enhao/EQUI-VOCAL/logs', udf_class, '{}_{}.log'.format(gt_udf_impl, log_name_suffix)), mode='w')
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    """
    Generate UDFs
    """
    if stage in ["udf_generation", "all"]:
        config = yaml.load(open("/home/enhao/EQUI-VOCAL/experiments/gpt/prompt.yaml", "r"), Loader=yaml.FullLoader)
        generate_semantic_interpretation_base_prompt = config["prompt"]["generate_semantic_interpretation"]
        generate_udfs_base_prompt = config["prompt"]["generate_udfs"]
        logger.info("Generating {} UDF interpretations".format(num_interpretations))
        generate_semantic_interpretation_prompt = replace_slot(generate_semantic_interpretation_base_prompt, {
            "num_interpretations": num_interpretations,
            "udf_description": udf_description,
            "obj_dictionary_str": "one object dictionary (o1)" if n_obj == 1 else "two object dictionaries (o1, o2)",
            "n_obj": "one object" if n_obj == 1 else "two objects",
            "udf_header": "py_{}({}, **kwargs)".format(udf_class, "o1" if n_obj == 1 else "o1, o2"),
        })
        interpretations = generate_semantic_interpretation(generate_semantic_interpretation_prompt)
        logger.info("Generating UDFs")

        for i in range(num_interpretations):
            logger.info("Generating UDF {}".format(i))
            generate_udfs_prompt = replace_slot(generate_udfs_base_prompt, {
                "udf_class": udf_class,
                "udf_description": udf_description,
                "semantic_interpretation": interpretations[i],
                "n_obj": "one object" if n_obj == 1 else "two objects",
                "udf_header": "py_{}({}, **kwargs)".format(udf_class, "o1" if n_obj == 1 else "o1, o2"),
            })
            generate_udf(i, generate_udfs_prompt, interpretations[i])

    """
    UDF selection
    """
    if stage in ["udf_selection", "all"]:
        logger.info("{} test samples in total".format(total_samples_test))
        if total_samples_test > 10000:
            logger.info("randomly sample 10000 samples from the test set")
            # randomly sample 10000 samples from the test set
            df_test = df_test.sample(n=10000, random_state=1)

        # Dynamically import the ground truth UDF
        module_name = "udfs.{}.{}".format(udf_class, gt_udf_impl)
        module = importlib.import_module(module_name)
        gt_udf = getattr(module, udf_generation_name)

        # Read UDF candidates from json files
        udf_candidates_with_scores = []
        for i in range(num_interpretations):
            with open(os.path.join("/home/enhao/EQUI-VOCAL/outputs/udf_generation", udf_class, "{}_{}.json".format(udf_class, i)), "r") as f:
                udf_candidate = json.load(f)
                if len(udf_candidate["kwargs_signature"]) == 0:
                    udf_candidates_with_scores.append([str(i), udf_candidate, 1, 0])
                elif num_parameter_search <= 0:
                    logger.info("use default values of kwargs in udf candidate {}".format(i))
                    try:
                        udf_id = str(i) + "_" + "_".join(["{}{}".format(k, v["default"]) for k, v in udf_candidate["kwargs_signature"].items()])
                    except Exception as e:
                        logger.debug("ERROR: failed to construct udf_id", e)
                        logger.debug("udf_candidate {}".format(udf_candidate))
                        udf_id = str(i)
                    udf_candidates_with_scores.append([udf_id, udf_candidate, 1, 0])
                else:
                    for _ in range(num_parameter_search):
                        # deepcopy udf_candidate
                        udf_candidate_variant = copy.deepcopy(udf_candidate)
                        for k, v in udf_candidate_variant["kwargs_signature"].items():
                            # randomly sample a value from the range
                            udf_candidate_variant["kwargs_signature"][k]["default"] = np.random.uniform(v["min"], v["max"])
                        # create a unique udf identifier for each udf candidate by concatenating the ksys and the default values of the kwargs
                        try:
                            udf_id = str(i) + "_" + "_".join(["{}{}".format(k, v["default"]) for k, v in udf_candidate_variant["kwargs_signature"].items()])
                        except Exception as e:
                            logger.debug("ERROR: failed to construct udf_id", e)
                            logger.debug("udf_candidate {}".format(udf_candidate))
                            udf_id = str(i)
                        udf_candidates_with_scores.append([udf_id, udf_candidate_variant, 1, 0])
            logger.info("udf {} implementation:\n{}".format(udf_candidates_with_scores[-1][0], udf_candidates_with_scores[-1][1]["function_implementation"]))

        # Select new video segments to label
        segment_selection_time = 0
        _start_segment_selection_time = time.time()
        for iter in range(labeling_budget):
            logger.info("iter {}".format(iter))
            _start_segment_selection_time_per_iter = time.time()
            # sort udf_candidates_with_scores by score
            udf_candidates_with_scores = sorted(udf_candidates_with_scores, key=lambda x: x[2], reverse=True)
            new_labeled_index = select_sample(udf_candidates_with_scores, udf_generation_name, df_train, n_obj)
            logger.info("pick next segments {}".format(new_labeled_index))
            labeled_index += new_labeled_index
            logger.info("# labeled segments {}".format(len(labeled_index)))
            if n_obj == 1:
                y_true = df_train.loc[labeled_index].apply(lambda row: gt_udf(row["o1"]), axis=1)
            elif n_obj == 2:
                y_true = df_train.loc[labeled_index].apply(lambda row: gt_udf(row["o1"], row["o2"]), axis=1)
            # log number of positive and negative samples
            logger.info("# positive: {}, # negative: {}".format(sum(y_true), len(y_true) - sum(y_true)))
            # Update scores
            updated_scores = []
            for _, udf_candidate, _, _ in udf_candidates_with_scores:
                updated_scores.append(compute_udf_score(gt_udf, udf_candidate, udf_generation_name, n_obj, df_train.loc[labeled_index], df_train.loc[new_labeled_index]))
            for i in range(len(udf_candidates_with_scores)):
                udf_candidates_with_scores[i][2] = updated_scores[i][0]
                udf_candidates_with_scores[i][3] += updated_scores[i][1]
            logger.info("updated udf_candidates_with_scores {}".format(["[udf_{}]: {}, {}".format(udf_id, score, loss_t) for udf_id, _, score, loss_t in udf_candidates_with_scores]))
            logger.info("test segment_selection_time_per_iter time: {}".format(time.time() - _start_segment_selection_time_per_iter))
        segment_selection_time += time.time() - _start_segment_selection_time
        logger.info("test segment_selection_time time: {}".format(segment_selection_time))

        # sort udf_candidates_with_scores by score
        udf_candidates_with_scores = sorted(udf_candidates_with_scores, key=lambda x: x[2], reverse=True)
        # logger.info("final udf_candidates_with_scores {}".format(["[udf_{}]: {}, {}".format(udf_id, score, loss_t) for udf_id, _, score, loss_t in udf_candidates_with_scores]))

        # compute test F1 score
        logger.info("compute test F1 score")
        for i in range(len(udf_candidates_with_scores)):
            f1_score_test = compute_udf_score(gt_udf, udf_candidates_with_scores[i][1], udf_generation_name, n_obj, df_test)
            logger.info("udf {}: test f1 {}, train f1 {}, n_misclassified {}".format(udf_candidates_with_scores[i][0], f1_score_test, udf_candidates_with_scores[i][2], udf_candidates_with_scores[i][3]))

        # compute the F1 score of the best udf (median F1 scores if there are multiple udfs with the same best score on the training set) on the test dataset
        best_score = udf_candidates_with_scores[0][2]
        best_candidates = [udf_candidates_with_scores[0]]
        for i in range(1, len(udf_candidates_with_scores)):
            if udf_candidates_with_scores[i][2] == best_score:
                best_candidates.append(udf_candidates_with_scores[i])
            else:
                break

        f1_score_test_list = []
        for best_candidate in best_candidates:
            f1_score_test = compute_udf_score(gt_udf, best_candidate[1], udf_generation_name, n_obj, df_test)
            f1_score_test_list.append(f1_score_test)
        median_f1_score_test = np.median(f1_score_test_list)
        logger.info("median test f1 {}".format(median_f1_score_test))




