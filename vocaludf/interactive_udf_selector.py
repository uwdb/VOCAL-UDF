import os
from enum import Enum
import time
import cv2
import duckdb
import logging
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score
import torch
import pandas as pd
import torchvision.ops as ops
import torchvision.transforms as T
from vocaludf import mlp
from vocaludf.utils import (
    parse_signature,
    transform_function,
    expand_box,
    SharedResources,
    UtilsMixin,
    UDFCandidate,
)
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio
from rich.prompt import Prompt
from rich.panel import Panel
from rich import print as rprint   # avoid clobbering built-in print

from vocaludf.udf_selector import UDFSelector

logger = logging.getLogger(__name__)

SamplingStrategy = Enum('SamplingStrategy', ['positive', 'negative', 'uncertainty'])

# --------------------------------------------------------------------------- #
# Interactive selector
# --------------------------------------------------------------------------- #
class InteractiveUDFSelector(UDFSelector):
    def _ask_binary_label(self, img_path: Path) -> bool:
        """
        Ask the user to label *img_path*.
            1 → positive / true
            0 → negative / false
            s → skip / unsure   (returns -1 so the caller can ignore it)

        Because a plain terminal can't render the image, we just tell the user
        where it lives (open it with your favourite viewer) and wait for input.
        """
        rprint(
            Panel(f"""
Please label the image. The subject is highlighted in red box, and the object is highlighted in blue box.
udf_signature: {self.udf_signature}
udf_description: {self.udf_description}
Open the image below in any viewer, then come back here and type [bold]1[/bold] (yes) or [bold]0[/bold] (no):
[underline]{img_path}[/underline]
        """))

        while True:
            ans = Prompt.ask("Label (1 = positive, 0 = negative)").strip()
            if ans == "1":
                return True
            if ans == "0":
                return False
            # if ans.lower() == "s":
            #     return -1
            rprint("[red]Please type 1 or 0.[/red]")


    def request_label(self, df: pd.DataFrame, n_obj: int) -> int:  # noqa: D401
        """
        Blockingly ask the human for a binary label for *row*.

        Returns
        -------
        int
            1 → positive / true
            0 → negative / false
           -1 → user skipped (caller decides what to do with that)
        """
        assert len(df) == 1, "InteractiveUDFSelector.request_label() expects a single-row DataFrame"
        row = df.iloc[0]
        vid = row['vid']
        fid = row['fid']
        frame = self.frame_processing_for_program(vid, fid)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_size = frame.shape[:2]
        # Draw bounding box on the frame and save it to a temporary file
        if n_obj == 1:
            x1, y1, x2, y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
        else:
            o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], image_size)
            o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], image_size)
            cv2.rectangle(frame, (o1_x1, o1_y1), (o1_x2, o1_y2), color=(0, 0, 255), thickness=1)
            cv2.rectangle(frame, (o2_x1, o2_y1), (o2_x2, o2_y2), color=(255, 0, 0), thickness=1)
        img_path = Path(self.shared_resources.interactive_labeling_dir) / f"frame_{vid}_{fid}.jpg"
        cv2.imwrite(str(img_path), frame)

        # We must hop into the asyncio loop because our helper is async.
        label: bool = self._ask_binary_label(img_path)
        return label


    def _select(self, gt_udf_name, udf_candidate_list, df_with_img_column):
        if len(udf_candidate_list) == 0:
            return None
        udf_signature = udf_candidate_list[0].udf_signature
        udf_name, udf_vars = parse_signature(udf_signature)
        n_obj = len(udf_vars)
        # Add a dummy UDF that always returns True
        udf_description = udf_candidate_list[0].udf_description
        self.udf_signature = udf_signature
        self.udf_description = udf_description
        dummy_udf = UDFCandidate(id='dummy', payload={'udf_name': udf_name, 'udf_signature': udf_signature, 'udf_description': udf_description, 'semantic_interpretation': 'dummy', 'function_implementation': f'def py_{udf_name}(*args, **kwargs):\n    return True\n'})
        udf_candidate_list.append(dummy_udf)

        # Construct training data and test data
        df_train, df_test = self.construct_train_and_test_data(n_obj, self.n_train_selection, self.n_test_selection, df_with_img_column=df_with_img_column)

        # Select new video segments to label
        # TODO: df_train and llm_positive_df may contain the same tuples
        labeled_index = []
        llm_positive_labeled_index = []
        llm_negative_labeled_index = []
        labeled_df = pd.DataFrame()
        y_true = []
        segment_selection_time = 0
        _start_segment_selection_time = time.time()

        sampling_strategy = SamplingStrategy.positive
        for iter in range(self.labeling_budget):
            logger.info("Iteration {}: {}".format(iter, sampling_strategy))
            _start_segment_selection_time_per_iter = time.time()

            if sampling_strategy == SamplingStrategy.positive and self.llm_positive_df is not None and len(self.llm_positive_df) > len(llm_positive_labeled_index):
                new_labeled_index = [len(llm_positive_labeled_index)]
                logger.debug("pick next segments from llm_positive_df {}".format(new_labeled_index))
                llm_positive_labeled_index += new_labeled_index
                new_labeled_df = self.llm_positive_df.iloc[new_labeled_index]
            elif sampling_strategy == SamplingStrategy.negative and self.llm_negative_df is not None and len(self.llm_negative_df) > len(llm_negative_labeled_index):
                new_labeled_index = [len(llm_negative_labeled_index)]
                logger.debug("pick next segments from llm_negative_df {}".format(new_labeled_index))
                llm_negative_labeled_index += new_labeled_index
                new_labeled_df = self.llm_negative_df.iloc[new_labeled_index]
            else:
                new_labeled_index = self.select_sample(
                    udf_candidate_list, udf_name, df_train, n_obj, labeled_index, sampling_strategy
                )
                logger.debug("pick next segments {}".format(new_labeled_index))
                labeled_index += new_labeled_index
                new_labeled_df = df_train.iloc[new_labeled_index]
            logger.debug("# labeled segments {}".format(len(set(llm_positive_labeled_index)) + len(set(llm_negative_labeled_index)) + len(set(labeled_index))))

            # Request labels on-the-fly
            label = self.request_label(new_labeled_df, n_obj)
            y_true.append(label)
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
        segment_selection_time += time.time() - _start_segment_selection_time
        logger.debug(
            "test segment_selection_time time: {}".format(segment_selection_time)
        )

        logger.info("Compute training F1 score")
        # compute the F1 score of the best udf (median F1 scores if there are multiple udfs with the same best score on the training set) on the test dataset
        if sum(y_true) == 0:
            logger.info("No positive samples are labeled. Returning the dummy UDF.")
            selected_udf_candidate = [udf_candidate for udf_candidate in udf_candidate_list if udf_candidate.id == "dummy"][0]
        else:
            # Compute final f1 score (without adding one)
            for i in range(len(udf_candidate_list)):
                try:
                    udf_candidate_list[i].score = self.compute_udf_score(
                        udf_candidate_list[i],
                        udf_name,
                        n_obj,
                        labeled_df,
                        y_true,
                    )
                except Exception as e:
                    logger.exception(f"ERROR: failed to compute final f1 score of UDFCandidate(id={udf_candidate_list[i].id}): {e}")
                    udf_candidate_list[i].score = -1
                    continue
            best_score = max(udf_candidate.score for udf_candidate in udf_candidate_list)
            best_candidates = [
                udf_candidate
                for udf_candidate in udf_candidate_list
                if udf_candidate.score == best_score
            ]

            f1_score_test_list = []
            for best_candidate in best_candidates:
                f1_score_test_list.append(best_candidate.test_score)
            median_f1_score_test = np.median(f1_score_test_list)
            logger.debug("median test f1: {}".format(median_f1_score_test))
            # TODO: If there are multiple best udfs, select the one with faster execution time?
            # If there are multiple best udfs, dummy UDF will be preferred
            selected_udf_candidate = best_candidates[-1]

        if selected_udf_candidate.id not in ["model", "dummy"]:
            # Transforms the function by removing **kwargs from the function signature and replacing the kwargs with the actual values
            selected_udf_candidate.function_implementation = transform_function(
                original_code=selected_udf_candidate.function_implementation,
                instantiation_dict=selected_udf_candidate.kwargs,
            )
        logger.info(f"[Selected]: {str(selected_udf_candidate)}")
        return selected_udf_candidate


    def compute_udf_score(
        self,
        udf_candidate,
        udf_name,
        n_obj,
        df,
        y_true,
        df_newly_labeled=None,
        label=None,
        add_one=False,
    ):
        """
        Compute the F1 score of the UDF candidate on the data (train or test), using the ground truth UDF as the label
        if df_newly_labeled is provided, also compute the number of misclassified samples of them (which is used to compute loss_t)
        """
        # make a copy of y_true to avoid modifying the original list
        y_true = y_true.copy()
        if udf_candidate.id == "model":
                best_ckpt = udf_candidate.function_implementation
                # logger.debug("df before predict_with_data: {}".format(df[["vid", "fid", "o1_oid"]].to_string()))
                y_pred = self.predict_with_data(df, best_ckpt, n_obj)
                # logger.debug(f"y_pred: {y_pred}")
                # logger.debug("df after predict_with_data: {}".format(df[["vid", "fid", "o1_oid"]].to_string()))
                if df_newly_labeled is not None:
                    y_pred_new = self.predict_with_data(df_newly_labeled, best_ckpt, n_obj)
                    # logger.debug(f"y_pred_new: {y_pred_new}")
        else:
            try:
                # For each sampled row in df, construct o1 and o2
                kwargs = {}
                for k, v in udf_candidate.kwargs.items():
                    kwargs[k] = float(v)
                py_func_name = "py_{}".format(udf_name)
                exec(udf_candidate.function_implementation, globals())
                udf_obj = globals()[py_func_name]
                y_pred = self.exec_udf_with_data(df, udf_obj, kwargs, n_obj, requires_no_error=False, timeout=max(60, len(df)*0.2))
                if df_newly_labeled is not None:
                    y_pred_new = self.exec_udf_with_data(df_newly_labeled, udf_obj, kwargs, n_obj, requires_no_error=False, timeout=max(60, len(df_newly_labeled)*0.2))
            except Exception as e:
                logger.exception("ERROR: failed to execute udf_candidate {}: {}".format(udf_candidate.id, e))
                raise

        if add_one:
            # Add one TP prediction to the model
            y_true.append(True)
            y_pred.append(True)
        score = f1_score(y_true, y_pred, zero_division=0.0)
        logger.debug("udf_candidate: {}, score: {}".format(udf_candidate.id, score))
        # logger.info("y_true: {}, y_pred: {}".format(y_true, y_pred))
        logger.debug("predicted positive: {}, predicted negative: {}".format(sum(y_pred), len(y_pred) - sum(y_pred)))
        logger.debug("positive: {}, negative: {}".format(sum(y_true), len(y_true) - sum(y_true)))

        # Compute y_true_new and num_misclassified
        if df_newly_labeled is not None:
            y_true_new = [label]
            num_misclassified = np.sum(np.array(y_true_new != y_pred_new) * 1)
            return score, num_misclassified
        else:
            return score
