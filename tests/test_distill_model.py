import argparse
import os
from vocaludf.utils import (
    replace_slot,
    parse_signature,
    PredImageDataset,
)
from vocaludf.udf_proposer import UDFProposer, UDFCandidate, CustomImageDataset
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from PIL import Image
import time
import resource
import duckdb
import logging
from openai import OpenAI
import re
from collections import defaultdict
import importlib
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import copy
import cv2
from tqdm import tqdm
import sys
from transformers import CLIPProcessor, CLIPModel, LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch
from torch.utils.data import Dataset
import base64
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import string
import lightning.pytorch as pl
from vocaludf import mlp
import torchvision.ops as ops
import torchvision.transforms as T
import yaml
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook, get_active_domain

tqdm.pandas()
client = OpenAI()

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

class TestDistillModel(UDFProposer):
    def _distill_model(self, udf_signature, udf_description, gt_udf_name):
        """
        gt_udf_name: ground truth UDF name. If provided, compute llm's TP, FP, TN, FN, f1 score and test the trained model.
        """
        self.attribute_df = self.conn.execute(f"SELECT * FROM {self.dataset}_attributes").df()
        self.relationship_df = self.conn.execute(f"SELECT * FROM {self.dataset}_relationships").df()

        self.llm_positive_df = None
        self.llm_negative_df = None

        # Initialization for model distillation
        udf_name, udf_vars = parse_signature(udf_signature)
        self.udf_class = udf_name.lower()
        self.n_obj = len(udf_vars)
        assert self.n_obj in [1, 2], "n_obj must be 1 or 2"
        self.udf_description = udf_description

        # module_name, function_name = gt_udf_name.split(".")
        # module_name = "udfs.{}".format(module_name)
        # module = importlib.import_module(module_name)
        # self.gt_udf = getattr(module, function_name)
        self.gt_udf_name = gt_udf_name

        # ask LLM about relevant object classes to the target relationships, and filter data
        filtered_objects, filtered_subjects, filtered_targets = None, None, None
        if self.dataset in ["charades"]:
            filtered_objects = list(set(self.llm_filter_relevant_objects(udf_signature, udf_description) + ['person']))
        elif self.dataset in ["gqa", "vaw"]:
            if self.n_obj == 1:
                filtered_objects = self.llm_filter_relevant_objects(udf_signature, udf_description)
            else: # n_obj == 2
                filtered_subjects, filtered_targets = self.llm_filter_relevant_subjects_targets(udf_signature, udf_description)
        logger.debug(f"filtered_objects: {filtered_objects}, filtered_subjects: {filtered_subjects}, filtered_targets: {filtered_targets}")

        num_active_learning_rounds = (self.n_train_distill - 1) // 100
        labeled_indices = set()
        for active_learning_round in range(num_active_learning_rounds + 1):
            logger.info(f"Active learning round: {active_learning_round}")
            if active_learning_round == 0:
                # NOTE: LLM doesn't generate labels in some cases, so we need to double the number of samples (i.e., self.n_train * 2) to ensure we have enough training samples
                self.df_train = self.construct_train_and_test_data(self.n_obj, int(min(self.n_train_distill, 100)), df_with_img_column=True, filtered_objects=filtered_objects, filtered_subjects=filtered_subjects, filtered_targets=filtered_targets)
                # self.df_train_unfiltered = self.construct_train_and_test_data(self.n_obj, int(min(self.n_train_distill, 100)*0.2), df_with_img_column=True)
                # self.df_train = pd.concat([self.df_train, self.df_train_unfiltered], ignore_index=True)
                # shuffle the training data
                # self.df_train = self.df_train.sample(frac=1).reset_index(drop=True)
            self.llm_annotate_data(active_learning_round=active_learning_round)
            self.mlp_prepare_data()
            best_ckpt = self.train(active_learning_round)

            checkpoint = torch.load(best_ckpt)
            hyper_parameters = checkpoint["hyper_parameters"]
            best_mlp_model = mlp.MLPProd(**hyper_parameters)
            best_mlp_model.load_state_dict(checkpoint["state_dict"])
            best_mlp_model.eval()
            best_mlp_model.to(self.device)

            pred_dataset = PredImageDataset(self.conn, self.n_obj, self.attribute_features_dir, self.relationship_features_dir)
            pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=4096, num_workers=self.num_workers, shuffle=False)

            rows = []
            predictions = []
            uncertainties = []
            with torch.no_grad():
                for row, feature in tqdm(pred_loader, file=sys.stdout):
                    feature = feature.to(self.device)
                    pred, uncertainty = best_mlp_model(feature)
                    rows.extend(row.tolist())
                    predictions.extend(pred.cpu().tolist())
                    uncertainties.extend(uncertainty.cpu().tolist())

            if self.n_obj == 1:
                check_df = pd.DataFrame(rows, columns=['vid', 'fid', 'oid'])
                check_df['aname'] = self.gt_udf_name
                result = check_df.merge(self.attribute_df, on=['vid', 'fid', 'oid', 'aname'], how='left', indicator=True)
                result = result.drop_duplicates(subset=['vid', 'fid', 'oid', 'aname'])
                result = result.rename(columns={"oid": "o1_oid"})
            else:
                check_df = pd.DataFrame(rows, columns=['vid', 'fid', 'oid1', 'oid2'])
                check_df['rname'] = self.gt_udf_name
                result = check_df.merge(self.relationship_df, on=['vid', 'fid', 'oid1', 'oid2', 'rname'], how='left', indicator=True)
                result = result.drop_duplicates(subset=['vid', 'fid', 'oid1', 'oid2', 'rname'])
                result = result.rename(columns={"oid1": "o1_oid", "oid2": "o2_oid"})
            result['label'] = (result['_merge'] == 'both').astype(int)
            result = result.reset_index(drop=True)
            labels = result['label'].tolist()

            # Compute F1 score
            f1 = f1_score(labels, predictions)
            logger.info(f"F1 score: {f1}")
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            logger.info(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            if self.dataset == "charades":
                result['pred'] = predictions
                result['uncertainty'] = uncertainties
                result_human_object = result[result["o1_oid"] == 0]
                labels_human_object = result_human_object["label"].tolist()
                predictions_human_object = result_human_object["pred"].tolist()
                f1_human_object = f1_score(labels_human_object, predictions_human_object)
                logger.info(f"[human-object only] F1 score: {f1_human_object}")
                tn_1, fp_1, fn_1, tp_1 = confusion_matrix(labels_human_object, predictions_human_object).ravel()
                logger.info(f"[human-object only] TP: {tp_1}, FP: {fp_1}, TN: {tn_1}, FN: {fn_1}")

            if active_learning_round < num_active_learning_rounds:
                # TODO: for "charades", only select the rows with the highest uncertainty for the human-object relationship
                # Active learning: select a batch of rows with the highest uncertainty that are not labeled
                selected_indices = np.argsort(-np.array(uncertainties))
                if self.dataset == "charades":
                    mask = (result['o1_oid'] == 0)
                    filtered_indices = set(result.index[mask].tolist())
                    selected_indices = [i for i in selected_indices if i in filtered_indices and i not in labeled_indices]
                else:
                    selected_indices = [i for i in selected_indices if i not in labeled_indices]
                selected_indices = selected_indices[:min(100, self.n_train_distill - 100 * active_learning_round)]
                # Random sampling:
                # selected_indices = np.random.choice(len(uncertainties), min(100, self.n_train_distill - 100 * active_learning_round), replace=False)
                labeled_indices.update(selected_indices)
                logger.info(f"labeled_indices: {sorted(labeled_indices)}, len(labeled_indices): {len(labeled_indices)}")
                if self.n_obj == 1:
                    columns = ['vid', 'fid', 'o1_oid']
                    df_source = self.one_object_df
                else:
                    columns = ['vid', 'fid', 'o1_oid', 'o2_oid']
                    df_source = self.two_objects_df
                selected_rows = result.iloc[selected_indices][columns]
                self.df_train = df_source.merge(selected_rows, on=columns, how='inner').reset_index(drop=True)
                self.df_train = self.df_train.drop_duplicates(subset=columns)
                self.df_train["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, self.df_train["vid"], self.df_train["fid"]), total=len(self.df_train), file=sys.stdout, desc="Processing frames"))

    def _distill_model_balanced(self, udf_signature, udf_description, gt_udf_name):
        """
        gt_udf_name: ground truth UDF name. If provided, compute llm's TP, FP, TN, FN, f1 score and test the trained model.
        """
        self.attribute_df = self.conn.execute(f"SELECT * FROM {self.dataset}_attributes").df()
        self.relationship_df = self.conn.execute(f"SELECT * FROM {self.dataset}_relationships").df()

        self.llm_positive_df = None
        self.llm_negative_df = None

        # Initialization for model distillation
        udf_name, udf_vars = parse_signature(udf_signature)
        self.udf_class = udf_name.lower()
        self.n_obj = len(udf_vars)
        assert self.n_obj in [1, 2], "n_obj must be 1 or 2"
        self.udf_description = udf_description

        self.gt_udf_name = gt_udf_name

        self.df_train, self.df_test = self.construct_balanced_data(self.n_obj, self.n_train_distill, self.n_test_distill)
        # self.df_train_unfiltered = self.construct_train_and_test_data(self.n_obj, int(min(self.n_train_distill, 100)*0.2), df_with_img_column=True)
        # self.df_train = pd.concat([self.df_train, self.df_train_unfiltered], ignore_index=True)
        # shuffle the training data
        # self.df_train = self.df_train.sample(frac=1).reset_index(drop=True)
        self.llm_annotate_data()
        self.mlp_prepare_data()
        best_ckpt = self.train()
        if gt_udf_name and hasattr(self, 'df_test'):
            self.test()

        checkpoint = torch.load(best_ckpt)
        hyper_parameters = checkpoint["hyper_parameters"]
        best_mlp_model = mlp.MLPProd(**hyper_parameters)
        best_mlp_model.load_state_dict(checkpoint["state_dict"])
        best_mlp_model.eval()
        best_mlp_model.to(self.device)

        pred_dataset = PredImageDataset(self.conn, self.n_obj, self.attribute_features_dir, self.relationship_features_dir)
        pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=4096, num_workers=self.num_workers, shuffle=False)

        rows = []
        predictions = []
        uncertainties = []
        with torch.no_grad():
            for row, feature in tqdm(pred_loader, file=sys.stdout):
                feature = feature.to(self.device)
                pred, uncertainty = best_mlp_model(feature)
                rows.extend(row.tolist())
                predictions.extend(pred.cpu().tolist())
                uncertainties.extend(uncertainty.cpu().tolist())

        if self.n_obj == 1:
            check_df = pd.DataFrame(rows, columns=['vid', 'fid', 'oid'])
            check_df['aname'] = self.gt_udf_name
            result = check_df.merge(self.attribute_df, on=['vid', 'fid', 'oid', 'aname'], how='left', indicator=True)
            result = result.drop_duplicates(subset=['vid', 'fid', 'oid', 'aname'])
            result = result.rename(columns={"oid": "o1_oid"})
        else:
            check_df = pd.DataFrame(rows, columns=['vid', 'fid', 'oid1', 'oid2'])
            check_df['rname'] = self.gt_udf_name
            result = check_df.merge(self.relationship_df, on=['vid', 'fid', 'oid1', 'oid2', 'rname'], how='left', indicator=True)
            result = result.drop_duplicates(subset=['vid', 'fid', 'oid1', 'oid2', 'rname'])
            result = result.rename(columns={"oid1": "o1_oid", "oid2": "o2_oid"})
        result['label'] = (result['_merge'] == 'both').astype(int)
        result = result.reset_index(drop=True)
        labels = result['label'].tolist()

        # Compute F1 score
        f1 = f1_score(labels, predictions)
        logger.info(f"F1 score: {f1}")
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        logger.info(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        if self.dataset == "charades":
            result['pred'] = predictions
            result['uncertainty'] = uncertainties
            result_human_object = result[result["o1_oid"] == 0]
            labels_human_object = result_human_object["label"].tolist()
            predictions_human_object = result_human_object["pred"].tolist()
            f1_human_object = f1_score(labels_human_object, predictions_human_object)
            logger.info(f"[human-object only] F1 score: {f1_human_object}")
            tn_1, fp_1, fn_1, tp_1 = confusion_matrix(labels_human_object, predictions_human_object).ravel()
            logger.info(f"[human-object only] TP: {tp_1}, FP: {fp_1}, TN: {tn_1}, FN: {fn_1}")

    def mlp_prepare_data(self):
        splits = ['train', 'test'] if self.gt_udf_name is not None else ['train']
        for split in splits:
            logger.info("Processing {} data".format(split))
            idx_to_remove = []
            for i in range(len(self.labeled_data[split])):
                if "image_features" in self.labeled_data[split][i]:
                    continue
                try:
                    data = self.labeled_data[split][i]
                    row = data['row']
                    logger.debug("row: {}".format(row.drop('img').to_dict()))
                    frame, image_size = self.frame_processing_for_model(row)
                    if frame is None: # failed to read the frame
                        idx_to_remove.append(i)
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_features = self.extract_features(frame, row, image_size)
                    self.labeled_data[split][i]["image_features"] = image_features
                except Exception as e:
                    logger.exception("Error: {}".format(e))
                    idx_to_remove.append(i)
                    continue
            for i in reversed(idx_to_remove):
                del self.labeled_data[split][i]

        # use 20% of the training data as validation data
        self.train_dataset = CustomImageDataset(self.labeled_data['train'], train=True)
        if self.gt_udf_name is not None:
            test_dataset = CustomImageDataset(self.labeled_data['test'], train=False)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    def train(self, active_learning_round=-1):
        # logger.debug("mlp_config: {}".format(mlp_config))
        mlp_dim_in = self.dim_in if self.n_obj == 1 else self.dim_in * 3
        logger.debug("mlp_dim_in: {}".format(mlp_dim_in)) # should be 512 for clip-vit-base-patch32
        self.checkpoint_root = os.path.join(self.config["model_dir"], "model_udf", self.dataset, f"{self.llm_method}_{self.mlp_method}", self.udf_class)
        if active_learning_round >= 0:
            self.checkpoint_root = os.path.join(self.checkpoint_root, f"active_learning_round_{active_learning_round}")

        # TODO: fine-tuning the learning rate
        # learningrate_callback = pl.callbacks.LearningRateFinder()
        # callbacks.append(learningrate_callback)

        best_model_score = float('inf')
        best_ckpt = None

        class_counts = [sum(data["llm_label"] == i for data in self.labeled_data['train']) for i in range(2)]
        try:
            self.class_weights = torch.tensor([1.0 / count for count in class_counts]).to(self.device)
        except ZeroDivisionError as e:
            logger.exception("Error: {}\nclass_counts: {}".format(e, class_counts))
            self.class_weights = torch.tensor([1.0, 1.0]).to(self.device)

        logger.debug("class_counts: {}, class_weights: {}".format(class_counts, self.class_weights))

        for i in range(10):
            logger.debug("Training model: trial {}".format(i))
            self.checkpoint_filename = "udf={}-run={}-ntrain={}-trial={}".format(self.udf_class, self.run_id, self.n_train_distill, i)
            os.makedirs(self.checkpoint_root, exist_ok=True)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filename=self.checkpoint_filename,
                monitor="val_loss",
                mode="min",
            )
            callbacks=[checkpoint_callback]
            earlystopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
            callbacks.append(earlystopping_callback)
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)

            train_set_size = int(len(self.train_dataset) * 0.8)
            valid_set_size = len(self.train_dataset) - train_set_size
            train_split, valid_split = torch.utils.data.random_split(self.train_dataset, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(self.run_id * 42 + i))

            self.train_loader = torch.utils.data.DataLoader(train_split, batch_size=512, shuffle=True)
            self.val_loader = torch.utils.data.DataLoader(valid_split, batch_size=512, shuffle=False)

            self.mlp_model = mlp.MLP(mlp_dim_in, 2, logger, self.class_weights) # binary classification

            self.trainer = pl.Trainer(
                # deterministic=self.deterministic,
                max_epochs=50,
                devices=1,
                accelerator="auto",
                enable_progress_bar=True,
                enable_checkpointing=True,
                enable_model_summary=False,
                # logger=pl_logger,
                default_root_dir=self.checkpoint_root,
                callbacks=callbacks,
                # check_val_every_n_epoch=5,
                # log_every_n_steps=min(50, len(dataset)-1),
                log_every_n_steps=1
            )

            self.trainer.fit(
                self.mlp_model,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.val_loader
            )

            # retrieve the best checkpoint after training
            current_model_score = checkpoint_callback.best_model_score
            logger.debug(f"current_model_score: {current_model_score}, best_model_score: {min(best_model_score, current_model_score)}")
            if current_model_score < best_model_score:
                best_model_score = current_model_score
                best_ckpt = checkpoint_callback.best_model_path
        logger.debug("Best model checkpoint: {}".format(best_ckpt))
        # best_mlp_model = mlp.MLP.load_from_checkpoint(best_ckpt)
        return best_ckpt

    def construct_balanced_data(self, n_obj, n_train=None, n_test=None):
        if self.dataset == "charades":
            return self._construct_train_and_test_data_with_images_person_object_relationships(n_obj, n_train, n_test)
        else:
            return self._construct_balanced_data_with_images(n_obj, n_train, n_test)

    def _construct_balanced_data_with_images(self, n_obj, n_train=None, n_test=None):
        if n_test:
            df_train, df_test = self._construct_balanced_data_without_images(n_obj, n_train, n_test)
            # construct the img column
            for df in [df_train, df_test]:
                df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
            return df_train, df_test
        else:
            df = self._construct_balanced_data_without_images(n_obj, n_train)
            df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
            return df

    def _construct_balanced_data_without_images(self, n_obj, n_train=None, n_test=None):
        # Construct training data and test data
        if n_obj == 1:
            pos_sql = """
                SELECT
                    o1.vid AS vid, o1.fid AS fid,
                    o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                    ARRAY[a1.aname] AS o1_gt_anames
                FROM {}_metadata m, {}_objects o1, {}_attributes a1
                WHERE m.vid = o1.vid AND m.fid = o1.fid AND o1.vid = a1.vid AND o1.fid = a1.fid AND o1.oid = a1.oid AND a1.aname = '{}'
                ORDER BY random()
                LIMIT {}
            """.format(
                self.dataset, self.dataset, self.dataset, self.udf_class, (n_train + n_test) // 2 if n_test else (n_train) // 2
            )
            neg_sql = """
                SELECT
                    o1.vid AS vid, o1.fid AS fid,
                    o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                    ARRAY[]::varchar[] as o1_gt_anames
                FROM {}_metadata m, {}_objects o1
                WHERE m.vid = o1.vid AND m.fid = o1.fid
                    AND NOT EXISTS (
                        SELECT 1
                        FROM {}_attributes a1
                        WHERE a1.vid = o1.vid AND a1.fid = o1.fid AND a1.oid = o1.oid AND a1.aname = '{}'
                    )
                ORDER BY random()
                LIMIT {}
            """.format(
                self.dataset, self.dataset, self.dataset, self.udf_class, (n_train + n_test) // 2 if n_test else (n_train) // 2
            )
        elif n_obj == 2:
            pos_sql = """
                SELECT
                    o1.vid AS vid, o1.fid AS fid,
                    o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                    o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2,
                    ARRAY[r.rname] as o1_o2_gt_rnames
                FROM {}_metadata m, {}_objects o1, {}_objects o2, {}_relationships r
                WHERE m.vid = o1.vid AND m.fid = o1.fid AND o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
                    AND o1.vid = r.vid AND o1.fid = r.fid AND o1.oid = r.oid1 AND o2.oid = r.oid2 AND r.rname = '{}'
                ORDER BY random()
                LIMIT {}
            """.format(
                self.dataset, self.dataset, self.dataset, self.dataset, self.udf_class, n_train // 2
            )
            neg_sql = """
                SELECT
                    o1.vid AS vid, o1.fid AS fid,
                    o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                    o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2,
                    ARRAY[]::varchar[] as o1_o2_gt_rnames
                FROM {}_metadata m, {}_objects o1, {}_objects o2
                WHERE m.vid = o1.vid AND m.fid = o1.fid AND o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
                    AND NOT EXISTS (
                        SELECT 1
                        FROM {}_relationships r
                        WHERE r.vid = o1.vid AND r.fid = o1.fid AND r.oid1 = o1.oid
                            AND r.oid2 = o2.oid AND r.rname = '{}'
                    )
                ORDER BY random()
                LIMIT {}
            """.format(
                self.dataset, self.dataset, self.dataset, self.dataset, self.udf_class, n_train // 2
            )
        else:
            raise ValueError("Number of objects not supported: {}".format(n_obj))

        # logger.debug("pos_sql: {}".format(pos_sql))
        # logger.debug("neg_sql: {}".format(neg_sql))
        df_pos = self.conn.execute(pos_sql).df()
        df_neg = self.conn.execute(neg_sql).df()

        # Interleave the rows of the two DataFrames
        df_filtered = pd.concat([df_pos, df_neg], ignore_index=True)
        max_length = max(len(df_pos), len(df_neg))
        new_index = np.array([[i, i + max_length] for i in range(max_length)]).flatten()
        new_index = new_index[new_index < len(df_filtered)]
        df_filtered = df_filtered.iloc[new_index]

        if n_test:
            df_train = df_filtered[:n_train]
            df_train = df_train.reset_index(drop=True)
            df_test = df_filtered[n_train:]
            df_test = df_test.reset_index(drop=True)
            return df_train, df_test
        else:
            df_filtered = df_filtered.reset_index(drop=True)
            return df_filtered

if __name__ == "__main__":

    # python test_distill_model.py --dataset "vaw" --udf_idx 0 --labeling_strategy "user" --n_train_distill 1000 --balanced
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--udf_idx", type=int, help="udf_idx")
    parser.add_argument("--labeling_strategy", type=str, choices=["user", "gpt4v", "llava"], default="user", help="labeling strategy for distill model annotations")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data")
    parser.add_argument("--balanced", action="store_true", help="use balanced dataset")

    args = parser.parse_args()
    dataset = args.dataset
    udf_idx = args.udf_idx
    labeling_strategy =args.labeling_strategy
    n_train_distill = args.n_train_distill
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data
    balanced = args.balanced

    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )

    if dataset == "charades":
        test_inputs = [
            ["holding(o0, o1)", "Whether o0 is holding o1.", "holding"],
            ["sitting_on(o0, o1)", "Whether o0 is sitting on o1.", "sitting_on"],
            ["standing_on(o0, o1)", "Whether o0 is standing on o1.", "standing_on"],
            ["covered_by(o0, o1)", "Whether o0 is covered by o1.", "covered_by"],
            ["carrying(o0, o1)", "Whether o0 is carrying o1.", "carrying"],
            ["eating(o0, o1)", "Whether o0 is eating o1.", "eating"],
            ["wiping(o0, o1)", "Whether object o0 is wiping o1", "wiping"],
            ["have_it_on_the_back(o0, o1)", "Whether o0 has o1 on the back.", "have_it_on_the_back"],
            ["leaning_on(o0, o1)", "Whether o0 is leaning on o1.", "leaning_on"],
            ["wearing(o0, o1)", "Whether o0 is wearing o1.", "wearing"],
        ]
    elif dataset == "vaw":
        test_inputs = [
            ["white(o0)", "Whether the color of o0 is white.", "white"],
            ["green(o0)", "Whether the color of o0 is green.", "green"],
            ["large(o0)", "Whether the size of o0 is large.", "large"],
            ["red(o0)", "Whether the color of o0 is red.", "red"],
            ["wooden(o0)", "Whether the material of o0 is wooden.", "wooden"],
            ["yellow(o0)", "Whether the color of o0 is yellow.", "yellow"],
            ["tall(o0)", "Whether the height of o0 is tall.", "tall"],
            ["silver(o0)", "Whether the color of o0 is silver.", "silver"],
            ["standing(o0)", "Whether the pose of o0 is standing.", "standing"],
            ["round(o0)", "Whether the shape of o0 is round.", "round"],
        ]
    test_input = test_inputs[udf_idx]
    """
    Set up logging
    """
    # Create a file handler that logs even debug messages
    log_dir = os.path.join(
        config["log_dir"],
        "distill_model",
        dataset,
        f"balanced={balanced}",
    )
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"labeling={labeling_strategy}-udf_idx={udf_idx}-ntrain_distill={n_train_distill}.log"), mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler with a higher log level
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    # logger.addHandler(console_handler)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    sys.excepthook = exception_hook

    registered_functions = [{
        "signature": "object(o0, name)",
        "description": "Whether o0 is an object with the given name.",
        "function_implementation": ""
    }]
    object_domain, relationship_domain, attribute_domain = get_active_domain(config, dataset, registered_functions)
    num_workers = 8
    up = TestDistillModel(
            config,
            prompt_config,
            registered_functions,
            object_domain,
            relationship_domain,
            attribute_domain,
            dataset,
            0,
            500,
            0,
            0,
            False,
            False,
            0,
            0,
            num_workers,
            save_labeled_data,
            load_labeled_data,
            n_train_distill,
            selection_strategy='model',
            selection_labels='none',
            allow_kwargs_in_udf=False,
            llm_method=labeling_strategy,
        )
    if balanced:
        up._distill_model_balanced(*test_input)
    else:
        up._distill_model(*test_input)