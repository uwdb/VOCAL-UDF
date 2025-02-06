import os
from enum import Enum
import time
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

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SamplingStrategy = Enum('SamplingStrategy', ['positive', 'negative', 'uncertainty'])

class UDFSelector(UtilsMixin):
    def __init__(self, shared_resources: SharedResources, llm_positive_df, llm_negative_df):
        # Shared resources
        self.shared_resources = shared_resources
        self.config = shared_resources.config
        self.dataset = shared_resources.dataset
        self.labeling_budget = shared_resources.labeling_budget
        self.n_selection_samples = shared_resources.n_selection_samples
        self.program_with_pixels = shared_resources.program_with_pixels
        self.run_id = shared_resources.run_id
        self.executor = shared_resources.executor
        self.n_train_selection = shared_resources.n_train_selection
        self.n_test_selection = shared_resources.n_test_selection
        self.attribute_features_dir = shared_resources.attribute_features_dir
        self.relationship_features_dir = shared_resources.relationship_features_dir
        self.one_object_df = shared_resources.one_object_df
        self.two_objects_df = shared_resources.two_objects_df
        self.device = shared_resources.device
        self.clip_model = shared_resources.clip_model
        self.tokenizer = shared_resources.tokenizer
        self.vid_to_vname = shared_resources.vid_to_vname

        # Per-UDF state variables
        self.llm_positive_df = llm_positive_df
        self.llm_negative_df = llm_negative_df
        self.cost_estimation = defaultdict(float)
        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )

    def get_cost_estimation(self):
        return self.cost_estimation


    ###########################
    ######               ######
    ###### UDF Selection ######
    ######               ######
    ###########################
    # TODO: If all the generated UDFs are terrible, we do rather not select any UDF (by using a dummy UDF that always returns True).
    # We could either add a dummy UDF to the candidate list,
    # or only register UDFs when the F1 score is above a certain threshold (how to decide the threshold?) to avoid generating terrible UDFs
    def select(self, gt_udf_name, udf_candidate_list):
        df_with_img_column = self.program_with_pixels
        for udf_candidate in udf_candidate_list:
            if udf_candidate.id == "model":
                df_with_img_column = True
                break
        return self._select(gt_udf_name, udf_candidate_list, df_with_img_column=df_with_img_column)


    def _select(self, gt_udf_name, udf_candidate_list, df_with_img_column):
        if len(udf_candidate_list) == 0:
            return None
        # if len(udf_candidate_list) == 1:
        #     selected_udf_candidate = udf_candidate_list[0]
        # else:
        udf_signature = udf_candidate_list[0].udf_signature
        udf_name, udf_vars = parse_signature(udf_signature)
        n_obj = len(udf_vars)
        # Add a dummy UDF that always returns True
        udf_description = udf_candidate_list[0].udf_description
        dummy_udf = UDFCandidate(id='dummy', payload={'udf_name': udf_name, 'udf_signature': udf_signature, 'udf_description': udf_description, 'semantic_interpretation': 'dummy', 'function_implementation': f'def py_{udf_name}(*args, **kwargs):\n    return True\n'})
        udf_candidate_list.append(dummy_udf)

        # Construct training data and test data
        df_train, df_test = self.construct_train_and_test_data(n_obj, self.n_train_selection, self.n_test_selection, df_with_img_column=df_with_img_column)

        # Select new video segments to label
        # TODO: df_train and llm_positive_df may contain the same tuples
        labeled_index = []
        llm_positive_labeled_index = []
        llm_negative_labeled_index = []
        segment_selection_time = 0
        _start_segment_selection_time = time.time()
        # TODO: perhaps regenerate one more UDF based on current labels after every k iterations

        sampling_strategy = SamplingStrategy.positive
        for iter in range(self.labeling_budget):
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
            if n_obj == 1:
                labeled_df_list = [df_train.iloc[labeled_index]['o1_gt_anames']]
                if self.llm_positive_df is not None and len(self.llm_positive_df):
                    labeled_df_list.append(self.llm_positive_df.iloc[llm_positive_labeled_index]['o1_gt_anames'])
                if self.llm_negative_df is not None and len(self.llm_negative_df):
                    labeled_df_list.append(self.llm_negative_df.iloc[llm_negative_labeled_index]['o1_gt_anames'])
                y_true = pd.Series([gt_udf_name in anames for anames in pd.concat(labeled_df_list)])
            elif n_obj == 2:
                labeled_df_list = [df_train.iloc[labeled_index]['o1_o2_gt_rnames']]
                if self.llm_positive_df is not None and len(self.llm_positive_df):
                    labeled_df_list.append(self.llm_positive_df.iloc[llm_positive_labeled_index]['o1_o2_gt_rnames'])
                if self.llm_negative_df is not None and len(self.llm_negative_df):
                    labeled_df_list.append(self.llm_negative_df.iloc[llm_negative_labeled_index]['o1_o2_gt_rnames'])
                logger.debug(f"pd.concat(labeled_df_list): {pd.concat(labeled_df_list)}")
                y_true = pd.Series([gt_udf_name in rnames for rnames in pd.concat(labeled_df_list)])
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
                labeled_df = [df_train.iloc[labeled_index]]
                if self.llm_positive_df is not None and len(self.llm_positive_df):
                    labeled_df.append(self.llm_positive_df.iloc[llm_positive_labeled_index])
                if self.llm_negative_df is not None and len(self.llm_negative_df):
                    labeled_df.append(self.llm_negative_df.iloc[llm_negative_labeled_index])
                labeled_df = pd.concat(labeled_df)
                try:
                    score, loss_t = self.compute_udf_score(
                        gt_udf_name,
                        udf_candidate_list[i],
                        udf_name,
                        n_obj,
                        labeled_df,
                        new_labeled_df,
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

        # compute test F1 score
        logger.info("compute test F1 score")
        for i in range(len(udf_candidate_list)):
            try:
                udf_candidate_list[i].test_score = self.compute_udf_score(
                    gt_udf_name,
                    udf_candidate_list[i],
                    udf_name,
                    n_obj,
                    df_test,
                )
                logger.info(str(udf_candidate_list[i]))
            except Exception as e:
                logger.exception(f"ERROR: failed to compute test F1 score of UDFCandidate(id={udf_candidate_list[i].id}): {e}")
                udf_candidate_list[i].test_score = -1
                continue

        logger.info("compute train F1 score")
        # compute the F1 score of the best udf (median F1 scores if there are multiple udfs with the same best score on the training set) on the test dataset
        if sum(y_true) == 0:
            logger.info("No positive samples are labeled. Returning the dummy UDF.")
            selected_udf_candidate = [udf_candidate for udf_candidate in udf_candidate_list if udf_candidate.id == "dummy"][0]
        else:
            # Compute final f1 score (without adding one)
            for i in range(len(udf_candidate_list)):
                labeled_df = [df_train.iloc[labeled_index]]
                if self.llm_positive_df is not None and len(self.llm_positive_df):
                    labeled_df.append(self.llm_positive_df.iloc[llm_positive_labeled_index])
                if self.llm_negative_df is not None and len(self.llm_negative_df):
                    labeled_df.append(self.llm_negative_df.iloc[llm_negative_labeled_index])
                labeled_df = pd.concat(labeled_df)
                try:
                    udf_candidate_list[i].score = self.compute_udf_score(
                        gt_udf_name,
                        udf_candidate_list[i],
                        udf_name,
                        n_obj,
                        labeled_df,
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
            logger.info("median test f1: {}".format(median_f1_score_test))
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


    def select_sample(
        self, udf_candidate_list, udf_name, df_train, n_obj, labeled_index, sampling_strategy
    ):


        prediction_matrix = []
        _start = time.time()

        # query_list = [query_graph.program for query_graph, _ in self.candidate_queries[:self.pool_size]]
        # logger.debug("query pool", [program_to_dsl(query, self.rewrite_variables) for query in query_list])
        unlabeled_index = np.setdiff1d(
            np.arange(len(df_train)), labeled_index, assume_unique=True
        )
        logger.debug("len(unlabeled_index): {}".format(len(unlabeled_index)))

        # sample a subset of videos during each iteration
        # If more than n_selection_samples videos, sample n_selection_samples videos
        if len(unlabeled_index) > self.n_selection_samples:
            sampled_index = np.random.choice(
                unlabeled_index, self.n_selection_samples, replace=False
            )
        else:
            sampled_index = unlabeled_index

        df_sampled = df_train.iloc[sampled_index]

        indices_to_remove = []
        for i, udf_candidate in enumerate(udf_candidate_list):
            logger.debug(f"Running udf_candidate(id={udf_candidate.id}) on sampled data")
            if udf_candidate.id == "model":
                # distilled-model UDF
                best_ckpt = udf_candidate.function_implementation
                predictions = self.predict_with_data(df_sampled, best_ckpt, n_obj)
                logger.debug("predictions: {}".format(predictions))
                prediction_matrix.append(predictions)
            else:
                try:
                    # program-based UDF
                    # For each sampled row in df_train, construct o1 and o2
                    kwargs = {}
                    for k, v in udf_candidate.kwargs.items():
                        kwargs[k] = float(v)
                    py_func_name = "py_{}".format(udf_name)
                    exec(udf_candidate.function_implementation, globals())
                    udf_obj = globals()[py_func_name]
                    # TODO: may need to timeout if running for too long
                    logger.debug(f"udf_name: {udf_name}")
                    result = self.exec_udf_with_data(df_sampled, udf_obj, kwargs, n_obj, requires_no_error=False, timeout=max(60, len(df_sampled)*0.2))
                    contains_non_boolean = False
                    for r in result:
                        if r != 1 and r != 0:
                            contains_non_boolean = True
                            break
                    if contains_non_boolean:
                        logger.debug(
                            f"ERROR: UDFCandidate(id={udf_candidate.id}) returned non-boolean value: {result}"
                        )
                        indices_to_remove.append(i)
                        continue
                except Exception as e:
                    logger.exception(f"ERROR: failed to execute UDFCandidate(id={udf_candidate.id}): {e}")
                    indices_to_remove.append(i)
                    continue
                prediction_matrix.append(result)
        # Remove UDFs that failed to execute
        for i in sorted(indices_to_remove, reverse=True):
            del udf_candidate_list[i]

        prediction_matrix = np.array(
            prediction_matrix
        ).transpose()  # (n_samples, n_udfs)
        logger.debug(
            "constructing prediction matrix took {} seconds".format(
                time.time() - _start
            )
        )
        logger.debug("prediction_matrix size {}".format(prediction_matrix.shape))

        # Reference: Learning Rare Category Classifiers on a Tight Labeling Budget
        # If the number of positives labeled so far is less than the number of negatives, we ask the human to label the most-likely positive images, otherwise we ask the human to label the images closest to the linear model’s margin
        eta_0 = np.sqrt(np.log(len(udf_candidate_list)) / 2)

        # Use F1-scores as weights
        posterior_t = [udf_candidate.score for udf_candidate in udf_candidate_list]
        # Use the original weights as in the paper
        # eta = eta_0 / np.sqrt(n_selection_samples)
        # loss_t = [loss_t for _, _, _, loss_t in udf_candidates_with_scores]
        # posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))

        posterior_t /= np.sum(posterior_t)  # normalized weight

        logger.debug("query weights {}".format(posterior_t))

        if sampling_strategy == SamplingStrategy.positive:
            # TODO: filter objects?
            # TODO: ask LLM?
            # find sample with highest weighted probability of being positive
            probability_list = np.zeros(len(sampled_index))
            for i in range(len(sampled_index)):
                probability_list[i] = np.inner(posterior_t, prediction_matrix[i, :])
            ind = np.argsort(-probability_list)
            logger.debug("probability list (desc): {}".format(probability_list[ind]))
            logger.debug("sampled index {}".format(sampled_index[ind]))
            max_probability_index = sampled_index[np.argmax(probability_list)]
            return [max_probability_index]
        elif sampling_strategy == SamplingStrategy.negative:
            probability_list = np.zeros(len(sampled_index))
            for i in range(len(sampled_index)):
                probability_list[i] = np.inner(posterior_t, prediction_matrix[i, :])
            ind = np.argsort(probability_list)
            logger.debug("probability list (asc): {}".format(probability_list[ind]))
            logger.debug("sampled index {}".format(sampled_index[ind]))
            min_probability_index = sampled_index[np.argmin(probability_list)]
            return [min_probability_index]
        else:
            entropy_list = np.zeros(len(sampled_index))
            for i in range(len(sampled_index)):
                entropy_list[i] = self._compute_u_t(posterior_t, prediction_matrix[i, :])
            ind = np.argsort(-entropy_list)
            logger.debug("entropy list {}".format(entropy_list[ind]))
            # df_object_pairs_train[sampled_index[ind]].apply(lambda row: logger.info("o1: {}, o2: {}".format(row["o1"], row["o2"])), axis=1)
            logger.debug("sampled index {}".format(sampled_index[ind]))
            # find argmax of entropy (top k)
            max_entropy_index = sampled_index[np.argmax(entropy_list)]
            return [max_entropy_index]

    def _compute_u_t(self, posterior_t, predictions_c):
        # Initialize possible u_t's
        u_t_list = np.zeros(2)

        # Repeat for each class
        for c in [0, 1]:
            # Compute the loss of models if the label of the streamed data is "c"
            loss_c = np.array(predictions_c != c) * 1
            # Compute the respective u_t value (conditioned on class c)
            term1 = np.inner(posterior_t, loss_c)
            u_t_list[c] = term1 * (1 - term1)

        # Return the final u_t
        u_t = np.max(u_t_list)

        return u_t

    def predict_with_data(self, df, ckpt, n_obj):
        # Predict the labels of all the data points
        if torch.cuda.is_available():
            checkpoint = torch.load(ckpt)
        else:
            checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
        hyper_parameters = checkpoint["hyper_parameters"]
        mlp_model = mlp.MLPProd(**hyper_parameters)
        mlp_model.load_state_dict(checkpoint["state_dict"])
        mlp_model.eval()
        mlp_model.to(self.device)

        # extract image features
        transforms = T.Compose([
            # T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            # T.CenterCrop(224),
            T.Lambda(lambda x: x / 255.0),        # Scale image data to [0, 1]
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        batch_size = 256
        if n_obj == 1:
            features, idxs_to_predict = self.extract_features_batch_one_object(df, transforms, batch_size)
        else:
            features, idxs_to_predict = self.extract_features_batch_two_objects(df, transforms, batch_size)
        batch_size = 65536
        predictions = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_data = features[i:i+batch_size]
                preds, _ = mlp_model(batch_data)
                # predictions.extend([bool(pred.item()) for pred in preds])
                predictions.extend(preds.cpu().tolist())

        all_predictions = [0] * len(df)
        for i, pred in zip(idxs_to_predict, predictions):
            all_predictions[i] = pred
        return all_predictions

    def extract_features_batch_one_object(self, df, transforms, batch_size):
        num_samples = len(df)
        all_features = []
        all_idxs_to_predict = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = batch_start + batch_size
            idxs_to_predict = []
            image_patches = []

            # Process each batch
            for i, (_, row) in enumerate(df.iloc[batch_start:batch_end].iterrows()):
                frame = row["img"]
                idxs_to_predict.append(batch_start + i)
                o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], (frame.shape[0], frame.shape[1]), factor=1)
                rois = [[0, o1_x1, o1_y1, o1_x2, o1_y2]]

                single_frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) # Shape: (1, C, H, W)

                rois_tensor = torch.tensor(rois, dtype=torch.float).to(self.device)
                # https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html
                # image_patches.shape: (K, C, 224, 224), where K is the number of bounding boxes
                patch_size=(224, 224)
                signle_frame_patches = ops.roi_align(single_frame, rois_tensor, output_size=patch_size, spatial_scale=1.0)
                image_patches.append(signle_frame_patches)

            image_patches = torch.cat(image_patches, dim=0)

            # Run CLIP model
            inputs = transforms(image_patches)
            with torch.no_grad():
                features = self.clip_model.get_image_features(pixel_values=inputs) # torch.FloatTensor of shape (batch_size, output_dim)

            all_features.append(features)
            all_idxs_to_predict.extend(idxs_to_predict)

        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)
        else:
            all_features = torch.tensor([], dtype=torch.float32)
        return all_features, all_idxs_to_predict

    def extract_features_batch_two_objects(self, df, transforms, batch_size):
        num_samples = len(df)
        all_features = []
        all_idxs_to_predict = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = batch_start + batch_size
            idxs_to_predict = []
            batch_boxes = []
            batch_o1_onames = []
            batch_o2_onames = []
            image_patches = []

            for i, (_, row) in enumerate(df.iloc[batch_start:batch_end].iterrows()):
                frame = row["img"]
                o1_x1, o1_y1, o1_x2, o1_y2 = expand_box(row['o1_x1'], row['o1_y1'], row['o1_x2'], row['o1_y2'], (frame.shape[0], frame.shape[1]))
                o2_x1, o2_y1, o2_x2, o2_y2 = expand_box(row['o2_x1'], row['o2_y1'], row['o2_x2'], row['o2_y2'], (frame.shape[0], frame.shape[1]))
                # Verify rois are correct
                roi_x1 = min(o1_x1, o2_x1)
                roi_y1 = min(o1_y1, o2_y1)
                roi_x2 = max(o1_x2, o2_x2)
                roi_y2 = max(o1_y2, o2_y2)
                if 0 <= roi_x1 < roi_x2 and 0 <= roi_y1 < roi_y2:
                    idxs_to_predict.append(batch_start + i)
                    rois = [[0, roi_x1, roi_y1, roi_x2, roi_y2]]
                    new_o1x1, new_o1y1, new_o1x2, new_o1y2, new_o2x1, new_o2y1, new_o2x2, new_o2y2 = self._compute_new_box_after_crop(row, (frame.shape[0], frame.shape[1]))
                    batch_boxes.append([int(new_o1x1), int(new_o1y1), int(new_o1x2), int(new_o1y2), int(new_o2x1), int(new_o2y1), int(new_o2x2), int(new_o2y2)])
                    batch_o1_onames.append(row['o1_oname'])
                    batch_o2_onames.append(row['o2_oname'])

                    single_frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) # Shape: (1, C, H, W)
                    rois_tensor = torch.tensor(rois, dtype=torch.float).to(self.device)
                    # https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html
                    # image_patches.shape: (K, C, 224, 224), where K is the number of bounding boxes
                    patch_size=(224, 224)
                    signle_frame_patches = ops.roi_align(single_frame, rois_tensor, output_size=patch_size, spatial_scale=1.0)
                    image_patches.append(signle_frame_patches)

            if len(image_patches) == 0:
                continue
            image_patches = torch.cat(image_patches, dim=0)
            # Run CLIP model
            batch_frames = image_patches.clone()
            # torch.tensor(image_patches).to(device)
            batch_boxes = torch.tensor(batch_boxes).to(self.device)
            N, C, H, W = batch_frames.shape
            # logger.debug(f"batch_frames.shape: {batch_frames.shape}, batch_boxes.shape: {batch_boxes.shape}")
            X = torch.arange(W, device=self.device).view(1, 1, W).expand(N, H, W)
            Y = torch.arange(H, device=self.device).view(1, H, 1).expand(N, H, W)
            subject_masks = (X >= batch_boxes[:, 0].view(N, 1, 1).expand(N, H, W)) & (X < batch_boxes[:, 2].view(N, 1, 1).expand(N, H, W)) & (Y >= batch_boxes[:, 1].view(N, 1, 1).expand(N, H, W)) & (Y < batch_boxes[:, 3].view(N, 1, 1).expand(N, H, W))
            target_masks = (X >= batch_boxes[:, 4].view(N, 1, 1).expand(N, H, W)) & (X < batch_boxes[:, 6].view(N, 1, 1).expand(N, H, W)) & (Y >= batch_boxes[:, 5].view(N, 1, 1).expand(N, H, W)) & (Y < batch_boxes[:, 7].view(N, 1, 1).expand(N, H, W))
            batch_frames_subject = batch_frames * subject_masks.unsqueeze(1).expand(N, C, H, W)
            batch_frames_target = batch_frames * target_masks.unsqueeze(1).expand(N, C, H, W)
            images = torch.cat([batch_frames, batch_frames_subject, batch_frames_target], dim=0) # (3N, C, H, W)
            inputs = transforms(images)
            if self.dataset == "charades":
                text_inputs = self.tokenizer(batch_o1_onames + batch_o2_onames, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(pixel_values=inputs) # torch.FloatTensor of shape (3N, 512)
                if self.dataset == "charades":
                    text_outputs = self.clip_model.get_text_features(**text_inputs) # torch.FloatTensor of shape (2N, 512)
            features = outputs.reshape(3, N, -1).permute(1, 0, 2).reshape(N, -1) # (N, 3 * 512)
            if self.dataset == "charades":
                text_features = text_outputs.reshape(2, N, -1).permute(1, 0, 2).reshape(N, -1) # (N, 2 * 512)
                features = torch.cat([features, text_features], dim=1) # (N, 5 * 512)
            all_features.append(features)
            all_idxs_to_predict.extend(idxs_to_predict)

        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)
        else:
            all_features = torch.tensor([], dtype=torch.float32)
        return all_features, all_idxs_to_predict

    # [[Deprecated]] Random access in Parquet is slow in general. Use predict_with_data instead.
    def predict_with_data_materialized(self, df, ckpt, n_obj):
        df = df.reset_index(drop=True)
        df['row_number'] = range(len(df))
        # Predict the labels of all the data points
        checkpoint = torch.load(ckpt)
        hyper_parameters = checkpoint["hyper_parameters"]
        mlp_model = mlp.MLPProd(**hyper_parameters)
        mlp_model.load_state_dict(checkpoint["state_dict"])
        mlp_model.eval()
        mlp_model.to(self.device)

        if n_obj == 1:
            df_with_features = self.conn.execute(f"""
                SELECT any_value(d.feature) as feature
                FROM df,
                    '{self.attribute_features_dir}/*.parquet' d
                WHERE df.vid=d.vid AND df.fid=d.fid
                    AND df.o1_oid=d.o1_oid
                GROUP BY df.row_number
                ORDER BY df.row_number
            """).df()
        else:
            df_with_features = self.conn.execute(f"""
                SELECT any_value(d.feature) as feature
                FROM df,
                    '{self.relationship_features_dir}/*.parquet' d
                WHERE df.vid=d.vid AND df.fid=d.fid
                    AND df.o1_oid=d.o1_oid AND df.o2_oid=d.o2_oid
                GROUP BY df.row_number
                ORDER BY df.row_number
            """).df()
        batch_size = 262144
        predictions = []
        with torch.no_grad():
            for i in range(0, len(df_with_features), batch_size):
                batch_data = df_with_features.iloc[i:i+batch_size]
                features = torch.tensor(batch_data["feature"].tolist(), dtype=torch.float32).to(self.device)
                preds = mlp_model(features)
                # predictions.extend([bool(pred.item()) for pred in preds])
                predictions.extend(preds.cpu().tolist())
        # predictions = []
        # with torch.no_grad():
        #     for _, row in tqdm(df_with_features.iterrows(), total=len(df_with_features), file=sys.stdout, desc="MLP Predicting"):
        #         feature = torch.tensor(row["feature"], dtype=torch.float32).to(self.device)
        #         pred = mlp_model(feature)
        #         predictions.append(bool(pred.item()))
        return predictions

    def compute_udf_score(
        self,
        gt_udf_name,
        udf_candidate,
        udf_name,
        n_obj,
        df,
        df_newly_labeled=None,
        add_one=False,
    ):
        """
        Compute the F1 score of the UDF candidate on the data (train or test), using the ground truth UDF as the label
        if df_newly_labeled is provided, also compute the number of misclassified samples of them (which is used to compute loss_t)
        """
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
                # y_pred = [False] * len(df)
                # if df_newly_labeled is not None:
                #     y_pred_new = [False] * len(df_newly_labeled)

        # Compute y_true and f1 score
        if n_obj == 1:
            # y_true = df.apply(lambda row: gt_udf_name in row["o1_gt_anames"], axis=1)
            y_true = [gt_udf_name in o1_gt_anames for o1_gt_anames in df['o1_gt_anames']]
        elif n_obj == 2:
            # y_true = df.apply(lambda row: gt_udf_name in row["o1_o2_gt_rnames"], axis=1)
            y_true = [gt_udf_name in o1_o2_gt_rnames for o1_o2_gt_rnames in df['o1_o2_gt_rnames']]
        # logger.debug(f"y_true: {y_true}, y_pred: {y_pred}")
        if add_one:
            # Add one TP prediction to the model
            y_true.append(True)
            y_pred.append(True)
        score = f1_score(y_true, y_pred, zero_division=0.0)
        logger.info("udf_candidate: {}, score: {}".format(udf_candidate.id, score))
        # logger.info("y_true: {}, y_pred: {}".format(y_true, y_pred))
        logger.info("predicted positive: {}, predicted negative: {}".format(sum(y_pred), len(y_pred) - sum(y_pred)))
        logger.info("positive: {}, negative: {}".format(sum(y_true), len(y_true) - sum(y_true)))

        # Compute y_true_new and num_misclassified
        if df_newly_labeled is not None:
            if n_obj == 1:
                # y_true_new = df_newly_labeled.apply(
                #     lambda row: gt_udf(row["o1_oname"], row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"], row["o1_anames"], row["height"], row["width"]), axis=1
                # )
                y_true_new = [gt_udf_name in o1_gt_anames for o1_gt_anames in df_newly_labeled['o1_gt_anames']]
            elif n_obj == 2:
                # y_true_new = df_newly_labeled.apply(
                #     lambda row: gt_udf(row["o1_oname"], row["o1_x1"], row["o1_y1"], row["o1_x2"], row["o1_y2"], row["o1_anames"], row["o2_oname"], row["o2_x1"], row["o2_y1"], row["o2_x2"], row["o2_y2"], row["o2_anames"], row["o1_o2_rnames"], row["o2_o1_rnames"], row["height"], row["width"]), axis=1
                # )
                y_true_new = [gt_udf_name in o1_o2_gt_rnames for o1_o2_gt_rnames in df_newly_labeled['o1_o2_gt_rnames']]
            # Count the number of misclassifications for the new samples
            # logger.debug(f"y_true_new: {y_true_new}, y_pred_new: {y_pred_new}")
            # logger.debug(f"y_true_new: {y_true_new}, y_pred_new: {y_pred_new}")
            num_misclassified = np.sum(np.array(y_true_new != y_pred_new) * 1)
            return score, num_misclassified
        else:
            return score
