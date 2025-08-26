from collections import defaultdict
import time
import pandas as pd
import yaml
import random
import json
import os
from vocaludf.utils import parse_signature, setup_logging, get_active_domain, SharedResources, UDFCandidate
import logging
import argparse
import numpy as np
from vocaludf.udf_selector import UDFSelector, SamplingStrategy
import duckdb
import sys
import resource
import asyncio
import ast
import re
import copy

# logging.basicConfig()
logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

project_root = os.getenv("PROJECT_ROOT")

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s GB'''%(point, usage/1024.0/1024.0 )

class ActiveUDFSelector(UDFSelector):
    def _select(self, gt_udf_name, udf_candidate_list):
        if len(udf_candidate_list) == 0:
            return None
        # if len(udf_candidate_list) == 1:
        #     selected_udf_candidate = udf_candidate_list[0]
        # else:
        self.udf_signature = udf_candidate_list[0].udf_signature
        udf_name, udf_vars = parse_signature(self.udf_signature)
        n_obj = len(udf_vars)
        # Add a dummy UDF that always returns True
        self.udf_description = udf_candidate_list[0].udf_description
        dummy_udf = UDFCandidate(id='dummy', payload={'udf_name': udf_name, 'udf_signature': self.udf_signature, 'udf_description': self.udf_description, 'semantic_interpretation': 'dummy', 'function_implementation': f'def py_{udf_name}(*args, **kwargs):\n    return True\n'})
        udf_candidate_list.append(dummy_udf)

        # Construct training data and test data
        df_train = self.construct_train_and_test_data(n_obj, n_train=self.n_train_selection, n_test=None, df_with_img_column=False)

        # Select new video segments to label
        # TODO: df_train and llm_positive_df may contain the same tuples
        labeled_index = []
        llm_positive_labeled_index = []
        llm_negative_labeled_index = []
        labeled_df = pd.DataFrame()
        y_true = []
        sampling_strategy = SamplingStrategy.positive
        iter = 0
        while sum(y_true) < 10:
            logger.info("iter {}: {}".format(iter, sampling_strategy))
            _start_segment_selection_time_per_iter = time.time()

            if sampling_strategy == SamplingStrategy.positive and self.llm_positive_df is not None and len(self.llm_positive_df) > len(llm_positive_labeled_index):
                new_labeled_index = [len(llm_positive_labeled_index)]
                logger.info("pick next segments from llm_positive_df {}".format(new_labeled_index))
                llm_positive_labeled_index += new_labeled_index
                new_labeled_df = self.llm_positive_df.iloc[new_labeled_index]
            elif sampling_strategy == SamplingStrategy.negative and self.llm_negative_df is not None and len(self.llm_negative_df) > len(llm_negative_labeled_index):
                new_labeled_index = [len(llm_negative_labeled_index)]
                logger.info("pick next segments from llm_negative_df {}".format(new_labeled_index))
                llm_negative_labeled_index += new_labeled_index
                new_labeled_df = self.llm_negative_df.iloc[new_labeled_index]
            else:
                new_labeled_index = self.select_sample(
                    udf_candidate_list, udf_name, df_train, n_obj, labeled_index, sampling_strategy
                )
                logger.info("pick next segments {}".format(new_labeled_index))
                labeled_index += new_labeled_index
                new_labeled_df = df_train.iloc[new_labeled_index]
            logger.info("# labeled segments {}".format(len(set(llm_positive_labeled_index)) + len(set(llm_negative_labeled_index)) + len(set(labeled_index))))

            # Request labels from either user or ground truth
            label = self.request_label(new_labeled_df, n_obj, gt_udf_name)
            y_true.append(label)
            # Load frame if df_with_img_column is True
            if self.df_with_img_column:
                new_labeled_df = new_labeled_df.copy()
                new_labeled_df["img"] = list(map(self.frame_processing_for_program, new_labeled_df["vid"], new_labeled_df["fid"]))
            labeled_df = pd.concat([labeled_df, new_labeled_df])
            # log number of positive and negative samples
            logger.info(
                "# positive: {}, # negative: {}".format(
                    sum(y_true), len(y_true) - sum(y_true)
                )
            )

            # Decide sampling strategy of next iteration
            if sum(y_true) <= len(y_true) - sum(y_true):
                sampling_strategy = SamplingStrategy.positive
            elif len(y_true) - sum(y_true) < 3:
                sampling_strategy = SamplingStrategy.negative
            else:
                sampling_strategy = SamplingStrategy.uncertainty

            # Update scores
            indices_to_remove = []
            for i in range(len(udf_candidate_list)):
                try:
                    score, loss_t = self.compute_udf_score(
                        udf_candidate_list[i],
                        udf_name,
                        n_obj,
                        labeled_df,
                        y_true,
                        new_labeled_df,
                        label,
                        add_one=True, # add one to avoid zero f1 score
                    )
                    udf_candidate_list[i].score = score
                    udf_candidate_list[i].loss_t += loss_t
                except Exception as e:
                    logger.exception(f"ERROR: failed to execute UDFCandidate(id={udf_candidate_list[i].id}): {e}")
                    indices_to_remove.append(i)
                    continue
            # Remove UDFs that failed to execute
            for i in sorted(indices_to_remove, reverse=True):
                del udf_candidate_list[i]
            # sort udf_candidate_list by score
            udf_candidate_list_sorted = sorted(
                udf_candidate_list, key=lambda x: x.score, reverse=True
            )
            logger.debug("updated udf_candidate_list: {}".format("\n".join([str(e) for e in udf_candidate_list_sorted])))
            logger.debug(
                "test segment_selection_time_per_iter time: {}".format(
                    time.time() - _start_segment_selection_time_per_iter
                )
            )
            iter += 1

        logger.info(
            "[{}] # active positive: {}, # active negative: {}, # active total: {}".format(
                gt_udf_name,
                sum(y_true), len(y_true) - sum(y_true), len(y_true)
            )
        )


def extract_udf_candidates(lines, udf_signature_to_gt_udf_name):
    results = {}
    gt_udf_name_to_best_ckpt = {}
    i = 0

    # Old-format single-line matches (backward compatible)
    old_info_re = re.compile(
        r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - .* - INFO - '
        r'\[(?P<sig>.*?)\] \[(?P<idx>\d+)\] (?P<field>\w+): (?P<val>.*)$'
    )
    old_ckpt_re = re.compile(
        r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - .* - INFO - '
        r'\[(?P<sig>.*?)\] Best model checkpoint: (?P<ckpt>.*)$'
    )

    # New-format anchors
    # Example header we key off of:
    # 2025-08-14 ... - vocaludf.async_udf_generator - INFO -
    new_block_header_re = re.compile(
        r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - .*async_udf_generator - INFO -\s*$'
    )
    # First non-empty line of block:
    # [holding(o0, o1)] [id=0]
    new_sig_id_re = re.compile(
        r'^\[(?P<sig>.+?)\]\s+\[id=(?P<idx>\d+)\]\s*$'
    )
    # Field headers inside block:
    # Semantic Interpretation: <one line text>
    new_semantic_re = re.compile(r'^Semantic Interpretation:\s*(?P<txt>.*)\s*$')
    # Function Implementation:  (code follows on subsequent lines)
    new_func_header_re = re.compile(r'^Function Implementation:\s*$')
    # Numeric Hyperparameters (kwargs): {...}   (possibly on one line)
    new_kwargs_inline_re = re.compile(
        r'^Numeric Hyperparameters\s*\(kwargs\):\s*(?P<val>.*)\s*$'
    )

    def _init_slot(sig, idx):
        if sig not in results:
            results[sig] = {}
        if idx not in results[sig]:
            results[sig][idx] = {}

    while i < len(lines):
        line = lines[i]

        # --- 1) Backward-compatible old single-line format ---
        m_old = old_info_re.match(line)
        if m_old:
            sig = m_old.group('sig')
            idx = int(m_old.group('idx'))
            field = m_old.group('field')
            val = m_old.group('val')
            _init_slot(sig, idx)

            if field == 'function_implementation':
                # capture subsequent non-timestamp lines as body
                func = val + '\n'
                i += 1
                while i < len(lines) and not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+', lines[i]):
                    func += lines[i] + '\n'
                    i += 1
                results[sig][idx][field] = func.rstrip('\n')
                continue
            elif field == 'kwargs':
                # capture multiline kwargs until next timestamp
                kwargs_str = val
                i += 1
                while i < len(lines) and not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+', lines[i]):
                    kwargs_str += '\n' + lines[i]
                    i += 1
                try:
                    kwargs_val = ast.literal_eval(kwargs_str)
                except Exception:
                    kwargs_val = {}
                results[sig][idx][field] = kwargs_val
                continue
            else:
                results[sig][idx][field] = val
                i += 1
                continue

        m_ckpt_old = old_ckpt_re.match(line)
        if m_ckpt_old:
            sig = m_ckpt_old.group('sig')
            ckpt = m_ckpt_old.group('ckpt').strip()
            if sig in udf_signature_to_gt_udf_name:
                gt = udf_signature_to_gt_udf_name[sig]
                gt_udf_name_to_best_ckpt[gt] = ckpt
            i += 1
            continue

        # --- 2) New block format ---
        # Look for the INFO header that introduces a block
        if new_block_header_re.match(line):
            # We expect the block to start within a few lines (skip blank lines)
            i += 1
            # Skip empty lines after header
            while i < len(lines) and lines[i].strip() == '':
                i += 1

            if i >= len(lines):
                break

            # Parse "[<signature>] [id=<n>]"
            m_sig = new_sig_id_re.match(lines[i])
            if not m_sig:
                # Not the UDF block we expected; move on.
                # (This keeps the parser tolerant to unrelated INFO blocks.)
                i += 1
                continue

            sig = m_sig.group('sig').strip()
            idx = int(m_sig.group('idx'))
            _init_slot(sig, idx)
            i += 1

            # Next expected lines inside the block (order is fixed per your example):
            # 1) Semantic Interpretation: <text>
            # (allow optional empty lines between sections)
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            if i < len(lines):
                m_sem = new_semantic_re.match(lines[i])
                if m_sem:
                    results[sig][idx]['semantic_interpretation'] = m_sem.group('txt').strip()
                    i += 1

            # 2) Function Implementation: \n <code... until kwargs line or blank+timestamp>
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            func_body = None
            if i < len(lines) and new_func_header_re.match(lines[i]):
                i += 1  # move to first code line
                func_lines = []
                while i < len(lines):
                    # Stop if we hit kwargs header, or a new timestamped log line, or a blank line
                    if new_kwargs_inline_re.match(lines[i]):
                        break
                    if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+', lines[i]):
                        break
                    # Allow empty lines within code
                    if lines[i].strip() == '' and (i + 1 < len(lines) and lines[i+1].strip() == ''):
                        # two consecutive blanks likely means block end; but keep one blank to be safe
                        pass
                    func_lines.append(lines[i])
                    i += 1
                func_body = '\n'.join(func_lines).rstrip('\n')
                if func_body:
                    results[sig][idx]['function_implementation'] = func_body

            # 3) Numeric Hyperparameters (kwargs): {...}
            if i < len(lines):
                m_kwargs = new_kwargs_inline_re.match(lines[i])
                if m_kwargs:
                    kwargs_raw = m_kwargs.group('val').strip()
                    # If kwargs spill to multiple lines (rare), capture until blank/timestamp
                    if kwargs_raw == '':
                        j = i + 1
                        cont = []
                        while j < len(lines):
                            if lines[j].strip() == '':
                                break
                            if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+', lines[j]):
                                break
                            cont.append(lines[j])
                            j += 1
                        kwargs_raw = '\n'.join(cont)
                        i = j
                    else:
                        i += 1

                    try:
                        kwargs_val = ast.literal_eval(kwargs_raw) if kwargs_raw else {}
                    except Exception:
                        kwargs_val = {}
                    results[sig][idx]['kwargs'] = kwargs_val

            # Done with this block; continue loop
            continue

        # Otherwise, nothing matched—advance.
        i += 1

    # Convert collected results into the desired mapping keyed by GT UDF name
    gt_udf_name_to_implemented_udfs = {}
    for udf_signature, idx_dict in results.items():
        # Skip signatures we don't have a GT mapping for
        if udf_signature not in udf_signature_to_gt_udf_name:
            continue
        idx_items = [idx_dict[k] for k in sorted(idx_dict.keys())]
        gt_name = udf_signature_to_gt_udf_name[udf_signature]
        gt_udf_name_to_implemented_udfs[gt_name] = idx_items

    return gt_udf_name_to_implemented_udfs, gt_udf_name_to_best_ckpt


async def main():
    # clevrer: python run_udf_selection_active_learning_count.py --num_missing_udfs 3 --query_id 2 --run_id 0 --dataset "clevrer" --query_filename "3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 100 --selection_strategy "both" --llm_method "gpt" --is_async --openai_model_name "gpt-4o"
    # cityflow: python run_udf_selection_active_learning_count.py --num_missing_udfs 2 --query_id 0 --run_id 0 --dataset "cityflow" --query_filename "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --llm_method "gpt" --is_async --openai_model_name "gpt-4o"
    # charades: python run_udf_selection_active_learning_count.py --num_missing_udfs 2 --query_id 3 --run_id 0 --dataset "charades" --query_filename "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --num_workers 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --llm_method "gpt" --is_async --openai_model_name "gpt-4o"
    config = yaml.safe_load(
        open(os.path.join(project_root, "configs", "config.yaml"), "r")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_missing_udfs", type=int, help="number of missing UDFs")
    parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--query_filename", type=str, help="query filename")
    parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in UDF")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument("--n_selection_samples", type=int, default=500, help="number of sampled videos per selection iteration")
    parser.add_argument("--ask_for_gt_udf", action="store_true", help="Ask for the gt_udf name interactively if enabled")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--program_with_pixels", action="store_true", help="program with pixels")
    parser.add_argument("--program_with_pretrained_models", action="store_true", help="program with pretrained models")
    parser.add_argument('--generate', action='store_true', help="only run the UDF generation step instead of actually executing the final query.")
    parser.add_argument("--num_workers", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--llm_method", type=str, choices=["gpt", "llava"], default="gpt", help="LLM method for distill model annotations")
    parser.add_argument("--is_async", action="store_true", help="use async for distilled-model UDF labeling")
    parser.add_argument("--openai_model_name", type=str, default="gpt-4-turbo-2024-04-09", help="OpenAI model name")

    args = parser.parse_args()
    num_missing_udfs = args.num_missing_udfs
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    query_filename = args.query_filename
    allow_kwargs_in_udf = args.allow_kwargs_in_udf
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    n_selection_samples = args.n_selection_samples
    ask_for_gt_udf = args.ask_for_gt_udf
    num_interpretations = args.num_interpretations
    program_with_pixels = args.program_with_pixels
    program_with_pretrained_models = args.program_with_pretrained_models
    if program_with_pretrained_models:
        assert program_with_pixels, "program_with_pretrained_models requires program_with_pixels"
    generate = args.generate
    num_workers = args.num_workers
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    llm_method = args.llm_method
    is_async = args.is_async
    openai_model_name = args.openai_model_name

    active_learning_config_name = "ninterp={}-nparams={}-kwargs={}-pixels={}-pretrained_models={}-ntrain_distill={}-nselection_samples={}-selection={}-budget={}-llm_method={}".format(
        num_interpretations,
        num_parameter_search,
        allow_kwargs_in_udf,
        program_with_pixels,
        program_with_pretrained_models,
        n_train_distill,
        n_selection_samples,
        selection_strategy,
        labeling_budget,
        llm_method,
    )

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_filename}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]

    # Set up logging
    base_dir = os.path.join(
        "udf_generation",
        dataset,
        query_filename,
        f"num_missing_udfs={num_missing_udfs}",
        f"active_learning_count_{active_learning_config_name}",
    )
    log_filename = "qid={}-run={}.log".format(query_id, run_id)
    setup_logging(config, base_dir, log_filename, logger)

    output_dir = os.path.join(
        config["output_dir"],
        base_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )
    registered_udfs_json = json.load(open(os.path.join(project_root, "vocaludf", "registered_udfs.json"), "r"))
    registered_functions = registered_udfs_json[f"{dataset}_base"]
    new_modules = input_query["new_modules"]
    assert num_missing_udfs >= 0 and num_missing_udfs <= len(new_modules), "num_missing_udfs must be between 0 and len(new_modules)"
    for new_module in new_modules[:(len(new_modules)-num_missing_udfs)]:
        registered_functions.append(registered_udfs_json[dataset][new_module])
    logger.info("Registered functions: {}".format(registered_functions))
    object_domain, relationship_domain, attribute_domain = get_active_domain(config, dataset, registered_functions)
    logger.info("Active domains: object={}, relationship={}, attribute={}".format(object_domain, relationship_domain, attribute_domain))

    with open(os.path.join(config['output_dir'], "udf_generation", dataset, query_filename, f"num_missing_udfs={num_missing_udfs}", active_learning_config_name, f"qid={query_id}-run={run_id}.json"), "r") as f:
        generation_output = json.load(f)

    # 1. Retrieve the UDF candidates from the log file
    with open(os.path.join(config['log_dir'], "udf_generation", dataset, query_filename, f"num_missing_udfs={num_missing_udfs}", active_learning_config_name, f"qid={query_id}-run={run_id}.log"), "r") as f:
        lines = f.readlines()
    has_udf_generation = False
    for line in lines:
        if "Shared resources initialization started" in line:
            has_udf_generation = True
            break
    if not has_udf_generation:
        # If the task doesn't contain UDF generation step, simply copy the generation output
        with open(os.path.join(output_dir, "qid={}-run={}.json".format(query_id, run_id)), "w") as f:
            json.dump(generation_output, f)
        return

    # retrieve gt_udf_name, implemented_udfs from the log file
    udf_signature_to_gt_udf_name = {}
    gt_udf_name_to_udf_signature = {}
    for i, line in enumerate(lines):
        if "Proposed functions:" in line:
            proposed_functions = ast.literal_eval(line.split("Proposed functions: ")[1])
            num_extracted_gt_udf_names = 0
            for j in range(i+1, len(lines)):
                if "Selected gt_udf_name:" in lines[j]:
                    udf_signature = list(proposed_functions.keys())[num_extracted_gt_udf_names]
                    gt_udf_name = lines[j].split("Selected gt_udf_name: ")[1].strip()
                    udf_signature_to_gt_udf_name[udf_signature] = gt_udf_name
                    gt_udf_name_to_udf_signature[gt_udf_name] = udf_signature
                    num_extracted_gt_udf_names += 1
                    if num_extracted_gt_udf_names == len(proposed_functions):
                        break
                elif "filtering out functions that are already registered" in lines[j]:
                    for k in range(j+1, len(lines)):
                        if "filtering out" in lines[k]:
                            filtered_out_udf_signature = lines[k].split("filtering out ")[1].strip()
                            del proposed_functions[filtered_out_udf_signature]
                        else:
                            break
            i = j + 1
            break
        i += 1
    gt_udf_name_to_implemented_udfs, gt_udf_name_to_best_ckpt = extract_udf_candidates(lines[i:], udf_signature_to_gt_udf_name)
    logger.info("gt_udf_name_to_implemented_udfs: {}".format(gt_udf_name_to_implemented_udfs))
    logger.info("gt_udf_name_to_best_ckpt: {}".format(gt_udf_name_to_best_ckpt))

    # 2. Initialize the shared resources
    logger.info("Shared resources initialization started")
    shared_resources = SharedResources(
        config,
        prompt_config,
        registered_functions,
        object_domain,
        relationship_domain,
        attribute_domain,
        dataset,
        labeling_budget,
        n_selection_samples,
        num_interpretations,
        num_parameter_search,
        program_with_pixels,
        program_with_pretrained_models,
        query_filename,
        query_id,
        run_id,
        num_workers,
        save_labeled_data,
        load_labeled_data,
        n_train_distill,
        selection_strategy,
        allow_kwargs_in_udf,
        llm_method,
        is_async,
        openai_model_name
    )
    logger.info("Shared resources initialization finished")

    # 3. Start the UDF selection process with random sampling
    logger.info("UDF selection started")
    for gt_udf_name, implemented_udfs in gt_udf_name_to_implemented_udfs.items():
        best_ckpt = gt_udf_name_to_best_ckpt[gt_udf_name]
        udf_signature = gt_udf_name_to_udf_signature[gt_udf_name]
        udf_description = proposed_functions[udf_signature]
        udf_name, udf_vars = parse_signature(udf_signature)
        # Prepare the UDF candidate list
        # Read UDF candidates from json files
        logger.info(f"Preparing UDF candidates for {udf_signature} started")
        udf_candidate_list = []  # List[UDFCandidate]
        for i in range(len(implemented_udfs)):
            try:
                udf_dict = implemented_udfs[i]
                udf_dict["udf_name"] = udf_name
                udf_dict["udf_signature"] = udf_signature
                udf_dict["udf_description"] = udf_description
                if allow_kwargs_in_udf and udf_dict.get("kwargs", {}):
                    # Instantiate the kwargs with default values
                    udf_variant_dict = copy.deepcopy(udf_dict)
                    udf_variant_dict["kwargs"] = {k: v["default"] for k, v in udf_variant_dict["kwargs"].items() if v["default"] is not None}
                    new_udf_candidate = UDFCandidate(id=i, payload=udf_variant_dict)
                    logger.debug(f"[{udf_signature}] {new_udf_candidate}")
                    udf_candidate_list.append(new_udf_candidate)
                    # Instantiate the kwargs with values randomly sampled from the range
                    if num_parameter_search and num_parameter_search > 0:
                        for _ in range(num_parameter_search):
                            # deepcopy udf_dict
                            udf_variant_dict = copy.deepcopy(udf_dict)
                            for k in list(udf_variant_dict["kwargs"].keys()):
                                # randomly sample a value from the range
                                udf_variant_dict["kwargs"][k] = np.random.uniform(udf_variant_dict["kwargs"][k]["min"], udf_variant_dict["kwargs"][k]["max"])
                            new_udf_candidate = UDFCandidate(id=i, payload=udf_variant_dict)
                            logger.debug(f"[{udf_signature}] {new_udf_candidate}")
                            udf_candidate_list.append(new_udf_candidate)
                else: # No additional arguments
                    new_udf_candidate = UDFCandidate(id=i, payload=udf_dict)
                    logger.debug(f"[{udf_signature}] {new_udf_candidate}")
                    udf_candidate_list.append(new_udf_candidate)
            except Exception as e:
                logger.exception(f"[{udf_signature}] Failed to read UDF candidate: {e}")
        udf_dict = {}
        udf_dict["udf_name"] = udf_name
        udf_dict["udf_signature"] = udf_signature
        udf_dict["udf_description"] = udf_description
        udf_dict["semantic_interpretation"] = 'model'
        udf_dict["function_implementation"] = best_ckpt
        new_udf_candidate = UDFCandidate(id='model', payload=udf_dict)
        udf_candidate_list.append(new_udf_candidate)
        logger.debug(f"[{udf_signature}] {new_udf_candidate}")
        logger.info(f"Preparing UDF candidates for {udf_signature} finished")

        logger.info(f"UDF selection for {udf_signature} started")
        us = ActiveUDFSelector(shared_resources, llm_positive_df=None, llm_negative_df=None)
        us.select(gt_udf_name, udf_candidate_list)
        logger.info(f"UDF selection for {udf_signature} finished")
    logger.info("UDF selection finished")

    logger.info("Peak memory usuage (in GB): {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0))

if __name__ == "__main__":
    asyncio.run(main())