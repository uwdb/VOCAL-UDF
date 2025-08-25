import argparse
import os
from vocaludf.utils import (
    replace_slot,
    parse_signature,
    PredImageDataset,
)
from vocaludf.async_udf_generator import UDFGenerator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from PIL import Image
import time
import resource
import duckdb
import logging
from openai import OpenAI, AsyncOpenAI
import re
from collections import defaultdict
import importlib
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import copy
import cv2
from tqdm import tqdm
import sys
from transformers import CLIPProcessor, CLIPModel, LlavaNextForConditionalGeneration, LlavaNextProcessor, CLIPTokenizer
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
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook, get_active_domain, SharedResources
import shutil
import asyncio

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

class LabelingQuality(UDFGenerator, SharedResources):
    def __init__(
        self,
        config,
        object_domain,
        relationship_domain,
        attribute_domain,
        dataset,
        num_workers,
        n_train_distill,
        llm_method,
        run_id,
        is_async,
        openai_model_name,
        udf_signature
    ):
        self.config = config
        self.object_domain = object_domain
        self.relationship_domain = relationship_domain
        self.attribute_domain = attribute_domain
        self.dataset = dataset
        self.num_workers = num_workers
        self.n_train_distill = n_train_distill
        self.llm_method = llm_method
        self.run_id = run_id
        self.is_async = is_async
        self.openai_model_name = openai_model_name
        self.udf_signature = udf_signature
        self.cost_estimation = defaultdict(float)
        self.execution_time = defaultdict(float)

        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.conn = duckdb.connect(
            database=os.path.join(self.config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

        self.init_table()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the CLIP model
        clip_model_name = os.path.join(self.config['model_dir'], 'clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.dim_in = self.clip_model.config.projection_dim

        if self.dataset == "charades":
            df_metadata = self.conn.execute(f"""
                SELECT DISTINCT vname, vid
                FROM charades_metadata
            """).df()
            self.vid_to_vname = {int(vid): vname for vid, vname in zip(df_metadata['vid'], df_metadata['vname'])}
        elif self.dataset == "cityflow":
            df_metadata = self.conn.execute(f"""
                SELECT vname, vid, fid
                FROM cityflow_metadata
            """).df()
            self.vid_to_vname = {(vid, fid): vname for vid, fid, vname in zip(df_metadata['vid'], df_metadata['fid'], df_metadata['vname'])}

    async def _label_data_balanced(self, gt_udf_name, udf_description, n_obj):
        self.n_obj = n_obj
        assert self.n_obj in [1, 2], "n_obj must be 1 or 2"

        self.gt_udf_name = gt_udf_name
        self.udf_description = udf_description

        self.df_train = self.construct_balanced_data(self.n_obj, self.n_train_distill)
        logger.info("df_train: {}".format(len(self.df_train)))

        await self.llm_annotate_data()

    async def llm_annotate_data(self, batch_size=8):
        labeled_data = defaultdict(list) # dictionary with 'train' and 'test' fields. Each field is a list of tuples (image_features, label)

        self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN = 0, 0, 0, 0
        # Training and validation data
        self.label_count = 0
        if self.llm_method == "gpt4v":
            if self.is_async:
                tasks = [asyncio.create_task(self.label_one(row, labeled_data)) for _, row in self.df_train.iterrows()]
                await asyncio.gather(*tasks)
            else:
                for _, row in self.df_train.iterrows():
                    logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++")
                    try:
                        gt_label = self._get_gt_label(row)
                        # Read and crop frame
                        logger.debug("row: {}".format(row.drop('img').to_dict()))
                        frame, image_size = self.frame_processing_for_model(row)
                        if frame is None:
                            continue
                        llm_label, base64_image, image_prompt = await self._llm_annotate_frame(frame, image_size, row, gt_label)
                        labeled_data['train'].append({"label": gt_label, "llm_label": llm_label, "base64_image": base64_image, "image_prompt": image_prompt, "row": row})
                    except Exception as e:
                        logger.exception("Error: {}".format(e))
                        continue
        elif self.llm_method == "llava":
            batched_rows = []
            batched_frames = []
            batched_image_sizes = []
            batched_gt_labels = []
            for _, row in self.df_train.iterrows():
                try:
                    # Read and crop frame
                    frame, image_size = self.frame_processing_for_model(row)
                    if frame is None:
                        continue
                    gt_label = self._get_gt_label(row)
                    batched_rows.append(row)
                    batched_frames.append(frame)
                    batched_image_sizes.append(image_size)
                    batched_gt_labels.append(gt_label)
                    if len(batched_rows) == batch_size:
                        generated_text, llm_labels, base64_images, image_prompts = self._llava_annotate_frame(batched_frames, batched_image_sizes, batched_rows, batched_gt_labels)
                        for i in range(len(batched_rows)):
                            if llm_labels[i] == -1:
                                continue
                            logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++")
                            logger.debug("row: {}".format(batched_rows[i].drop('img').to_dict()))
                            logger.debug("base64_image: {}".format(base64_images[i]))
                            logger.debug("Llava image prompt: {}".format(image_prompts[i]))
                            logger.debug("Llava result: {}".format(generated_text[i]))
                            logger.debug("gt_label: {}".format(batched_gt_labels[i]))
                            labeled_data['train'].append({"label": batched_gt_labels[i], "llm_label": llm_labels[i], "base64_image": base64_images[i], "image_prompt": image_prompts[i], "row": batched_rows[i]})
                        batched_rows = []
                        batched_frames = []
                        batched_image_sizes = []
                        batched_gt_labels = []
                except Exception as e:
                    logger.exception("Error: {}".format(e))
                    continue
        elif self.llm_method == "user": # Ground truth labels
            for _, row in self.df_train.iterrows():
                logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++")
                try:
                    gt_label = self._get_gt_label(row)
                    # Read and crop frame
                    logger.debug("row: {}".format(row.drop('img').to_dict()))
                    logger.debug("gt_label: {}".format(gt_label))
                    if gt_label == 1:
                        self.llm_TP += 1
                    else:
                        self.llm_TN += 1
                    self.label_count += 1
                    labeled_data['train'].append({"label": gt_label, "llm_label": gt_label, "base64_image": "", "image_prompt": "", "row": row})
                except Exception as e:
                    logger.exception("Error: {}".format(e))
                    continue
        if self.gt_udf_name is not None:
            llm_f1 = 2*self.llm_TP/(2*self.llm_TP+self.llm_FP+self.llm_FN) if 2*self.llm_TP+self.llm_FP+self.llm_FN > 0 else 0.0
            logger.debug("llm_TP: {}, llm_FP: {}, llm_TN: {}, llm_FN: {}, llm_f1: {}".format(self.llm_TP, self.llm_FP, self.llm_TN, self.llm_FN, llm_f1))
        else:
            llm_f1 = -1
        labeled_data["metadata"] = {"llm_TP": self.llm_TP, "llm_FP": self.llm_FP, "llm_TN": self.llm_TN, "llm_FN": self.llm_FN, "llm_f1": llm_f1}

    async def label_one(self, row, labeled_data):
        log_msgs = []
        log_msgs.append(f"[{self.udf_signature}] +++++++++++++++++++++++++++++++++++++++++++++++")
        # logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++")

        try:
            gt_label = self._get_gt_label(row)
            # Read and crop frame
            log_msgs.append("[{}] row: {}".format(self.udf_signature, row.drop('img').to_dict()))
            # logger.debug("row: {}".format(row.drop('img').to_dict()))
            frame, image_size = self.frame_processing_for_model(row)
            if frame is None:
                logger.debug("\n".join(log_msgs))
                return
            llm_label, base64_image, image_prompt = await self._async_llm_annotate_frame(frame, image_size, row, gt_label, log_msgs)
        except Exception as e:
            logger.debug("\n".join(log_msgs))
            logger.exception(f"[{self.udf_signature}] Error: {e}")
            return

        logger.debug("\n".join(log_msgs))
        labeled_data['train'].append({"label": gt_label, "llm_label": llm_label, "base64_image": base64_image, "image_prompt": image_prompt, "row": row})

    #### Unused ####
    # def _distill_model(self, udf_signature, udf_description, gt_udf_name):
    #     """
    #     gt_udf_name: ground truth UDF name. If provided, compute llm's TP, FP, TN, FN, f1 score and test the trained model.
    #     """
    #     self.attribute_df = self.conn.execute(f"SELECT * FROM {self.dataset}_attributes").df()
    #     self.relationship_df = self.conn.execute(f"SELECT * FROM {self.dataset}_relationships").df()

    #     self.llm_positive_df = None
    #     self.llm_negative_df = None

    #     # Initialization for model distillation
    #     udf_name, udf_vars = parse_signature(udf_signature)
    #     self.udf_class = udf_name.lower()
    #     self.n_obj = len(udf_vars)
    #     assert self.n_obj in [1, 2], "n_obj must be 1 or 2"
    #     self.udf_description = udf_description

    #     # module_name, function_name = gt_udf_name.split(".")
    #     # module_name = "udfs.{}".format(module_name)
    #     # module = importlib.import_module(module_name)
    #     # self.gt_udf = getattr(module, function_name)
    #     self.gt_udf_name = gt_udf_name

    #     # ask LLM about relevant object classes to the target relationships, and filter data
    #     filtered_objects, filtered_subjects, filtered_targets = None, None, None
    #     if self.dataset in ["charades"]:
    #         filtered_objects = list(set(self.llm_filter_relevant_objects(udf_signature, udf_description) + ['person']))
    #     elif self.dataset in ["gqa", "vaw"]:
    #         if self.n_obj == 1:
    #             filtered_objects = self.llm_filter_relevant_objects(udf_signature, udf_description)
    #         else: # n_obj == 2
    #             filtered_subjects, filtered_targets = self.llm_filter_relevant_subjects_targets(udf_signature, udf_description)
    #     logger.debug(f"filtered_objects: {filtered_objects}, filtered_subjects: {filtered_subjects}, filtered_targets: {filtered_targets}")

    #     num_active_learning_rounds = (self.n_train_distill - 1) // 100
    #     labeled_indices = set()
    #     for active_learning_round in range(num_active_learning_rounds + 1):
    #         logger.info(f"Active learning round: {active_learning_round}")
    #         if active_learning_round == 0:
    #             # NOTE: LLM doesn't generate labels in some cases, so we need to double the number of samples (i.e., self.n_train * 2) to ensure we have enough training samples
    #             self.df_train = self.construct_train_and_test_data(self.n_obj, int(min(self.n_train_distill, 100)), df_with_img_column=True, filtered_objects=filtered_objects, filtered_subjects=filtered_subjects, filtered_targets=filtered_targets)
    #             # self.df_train_unfiltered = self.construct_train_and_test_data(self.n_obj, int(min(self.n_train_distill, 100)*0.2), df_with_img_column=True)
    #             # self.df_train = pd.concat([self.df_train, self.df_train_unfiltered], ignore_index=True)
    #             # shuffle the training data
    #             # self.df_train = self.df_train.sample(frac=1).reset_index(drop=True)
    #         self.llm_annotate_data(active_learning_round=active_learning_round)
    #         self.mlp_prepare_data()
    #         best_ckpt = self.train(active_learning_round)

    #         checkpoint = torch.load(best_ckpt)
    #         hyper_parameters = checkpoint["hyper_parameters"]
    #         best_mlp_model = mlp.MLPProd(**hyper_parameters)
    #         best_mlp_model.load_state_dict(checkpoint["state_dict"])
    #         best_mlp_model.eval()
    #         best_mlp_model.to(self.device)

    #         pred_dataset = PredImageDataset(self.conn, self.n_obj, self.attribute_features_dir, self.relationship_features_dir)
    #         pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=4096, num_workers=self.num_workers, shuffle=False)

    #         rows = []
    #         predictions = []
    #         uncertainties = []
    #         with torch.no_grad():
    #             for row, feature in tqdm(pred_loader, file=sys.stdout):
    #                 feature = feature.to(self.device)
    #                 pred, uncertainty = best_mlp_model(feature)
    #                 rows.extend(row.tolist())
    #                 predictions.extend(pred.cpu().tolist())
    #                 uncertainties.extend(uncertainty.cpu().tolist())

    #         if self.n_obj == 1:
    #             check_df = pd.DataFrame(rows, columns=['vid', 'fid', 'oid'])
    #             check_df['aname'] = self.gt_udf_name
    #             result = check_df.merge(self.attribute_df, on=['vid', 'fid', 'oid', 'aname'], how='left', indicator=True)
    #             result = result.drop_duplicates(subset=['vid', 'fid', 'oid', 'aname'])
    #             result = result.rename(columns={"oid": "o1_oid"})
    #         else:
    #             check_df = pd.DataFrame(rows, columns=['vid', 'fid', 'oid1', 'oid2'])
    #             check_df['rname'] = self.gt_udf_name
    #             result = check_df.merge(self.relationship_df, on=['vid', 'fid', 'oid1', 'oid2', 'rname'], how='left', indicator=True)
    #             result = result.drop_duplicates(subset=['vid', 'fid', 'oid1', 'oid2', 'rname'])
    #             result = result.rename(columns={"oid1": "o1_oid", "oid2": "o2_oid"})
    #         result['label'] = (result['_merge'] == 'both').astype(int)
    #         result = result.reset_index(drop=True)
    #         labels = result['label'].tolist()

    #         # Compute F1 score
    #         f1 = f1_score(labels, predictions)
    #         logger.info(f"F1 score: {f1}")
    #         tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    #         logger.info(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    #         if self.dataset == "charades":
    #             result['pred'] = predictions
    #             result['uncertainty'] = uncertainties
    #             result_human_object = result[result["o1_oid"] == 0]
    #             labels_human_object = result_human_object["label"].tolist()
    #             predictions_human_object = result_human_object["pred"].tolist()
    #             f1_human_object = f1_score(labels_human_object, predictions_human_object)
    #             logger.info(f"[human-object only] F1 score: {f1_human_object}")
    #             tn_1, fp_1, fn_1, tp_1 = confusion_matrix(labels_human_object, predictions_human_object).ravel()
    #             logger.info(f"[human-object only] TP: {tp_1}, FP: {fp_1}, TN: {tn_1}, FN: {fn_1}")

    #         if active_learning_round < num_active_learning_rounds:
    #             # TODO: for "charades", only select the rows with the highest uncertainty for the human-object relationship
    #             # Active learning: select a batch of rows with the highest uncertainty that are not labeled
    #             selected_indices = np.argsort(-np.array(uncertainties))
    #             if self.dataset == "charades":
    #                 mask = (result['o1_oid'] == 0)
    #                 filtered_indices = set(result.index[mask].tolist())
    #                 selected_indices = [i for i in selected_indices if i in filtered_indices and i not in labeled_indices]
    #             else:
    #                 selected_indices = [i for i in selected_indices if i not in labeled_indices]
    #             selected_indices = selected_indices[:min(100, self.n_train_distill - 100 * active_learning_round)]
    #             # Random sampling:
    #             # selected_indices = np.random.choice(len(uncertainties), min(100, self.n_train_distill - 100 * active_learning_round), replace=False)
    #             labeled_indices.update(selected_indices)
    #             logger.info(f"labeled_indices: {sorted(labeled_indices)}, len(labeled_indices): {len(labeled_indices)}")
    #             if self.n_obj == 1:
    #                 columns = ['vid', 'fid', 'o1_oid']
    #                 df_source = self.one_object_df
    #             else:
    #                 columns = ['vid', 'fid', 'o1_oid', 'o2_oid']
    #                 df_source = self.two_objects_df
    #             selected_rows = result.iloc[selected_indices][columns]
    #             self.df_train = df_source.merge(selected_rows, on=columns, how='inner').reset_index(drop=True)
    #             self.df_train = self.df_train.drop_duplicates(subset=columns)
    #             self.df_train["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, self.df_train["vid"], self.df_train["fid"]), total=len(self.df_train), file=sys.stdout, desc="Processing frames"))

    # def _distill_model_balanced(self, gt_udf_name, n_obj):
    #     clip_model_name = os.path.join(self.config['model_dir'], 'clip-vit-base-patch32')
    #     self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    #     # Initialization for model distillation
    #     self.n_obj = n_obj
    #     assert self.n_obj in [1, 2], "n_obj must be 1 or 2"

    #     self.gt_udf_name = gt_udf_name

    #     self.df_train, self.df_val, self.df_test = self.construct_balanced_data(self.n_obj, self.max_positive_samples)
    #     logger.info("df_train: {}, df_val: {}, df_test: {}".format(len(self.df_train), len(self.df_val), len(self.df_test)))

    #     self.llm_annotate_data()
    #     self.mlp_prepare_data()
    #     best_ckpt = self.train()

    #     return best_ckpt



    # def mlp_prepare_data(self):
    #     splits = ['train', 'val', 'test']
    #     for split in splits:
    #         logger.info("Processing {} data".format(split))
    #         idx_to_remove = []
    #         for i in tqdm(range(len(self.labeled_data[split])), total=len(self.labeled_data[split]), file=sys.stdout, desc="mlp_prepare_data"):
    #             try:
    #                 data = self.labeled_data[split][i]
    #                 row = data['row']
    #                 frame, image_size = self.frame_processing_for_model(row)
    #                 if frame is None: # failed to read the frame
    #                     idx_to_remove.append(i)
    #                     continue
    #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 image_features = self.extract_features(frame, row, image_size)
    #                 if self.dataset == "charades":
    #                     text_features = self.extract_text_features(row)
    #                     self.labeled_data[split][i]["image_features"] = torch.cat([image_features, text_features], dim=-1)
    #                 else:
    #                     self.labeled_data[split][i]["image_features"] = image_features
    #             except Exception as e:
    #                 logger.exception("Error: {}".format(e))
    #                 idx_to_remove.append(i)
    #                 continue
    #         for i in reversed(idx_to_remove):
    #             del self.labeled_data[split][i]

    #     train_dataset = CustomImageDataset(self.labeled_data['train'], train=True)
    #     val_dataset = CustomImageDataset(self.labeled_data['val'], train=True)
    #     test_dataset = CustomImageDataset(self.labeled_data['test'], train=True)

    #     self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    #     self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
    #     self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    # def extract_text_features(self, row):
    #     if self.n_obj == 1:
    #         text = row["o1_oname"]
    #         inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
    #         with torch.no_grad():
    #             outputs = self.clip_model.get_text_features(**inputs)
    #         outputs = outputs.squeeze(0)
    #     else:
    #         inputs = self.tokenizer([row["o1_oname"], row["o2_oname"]], padding=True, return_tensors="pt").to(self.device)
    #         with torch.no_grad():
    #             outputs = self.clip_model.get_text_features(**inputs) # 2 x 512
    #         outputs = outputs.reshape(-1)

    #     return outputs

    # def train(self):
    #     # logger.debug("mlp_config: {}".format(mlp_config))
    #     # mlp_dim_in = self.dim_in if self.n_obj == 1 else self.dim_in * 3
    #     if self.dataset == "charades":
    #         mlp_dim_in = self.dim_in * 5
    #     elif self.dataset == "cityflow":
    #         mlp_dim_in = self.dim_in
    #     else:
    #         raise NotImplementedError
    #     logger.debug("mlp_dim_in: {}".format(mlp_dim_in)) # should be 512 for clip-vit-base-patch32
    #     self.checkpoint_root = os.path.join(self.config["model_dir"], "model_udf", self.dataset, self.gt_udf_name)

    #     best_model_score = float('inf')
    #     best_ckpt = None

    #     class_counts = [sum(data["llm_label"] == i for data in self.labeled_data['train']) for i in range(2)]
    #     try:
    #         self.class_weights = torch.tensor([1.0 / count for count in class_counts]).to(self.device)
    #     except ZeroDivisionError as e:
    #         logger.exception("Error: {}\nclass_counts: {}".format(e, class_counts))
    #         self.class_weights = torch.tensor([1.0, 1.0]).to(self.device)

    #     logger.debug("class_counts: {}, class_weights: {}".format(class_counts, self.class_weights))

    #     for i in range(10):
    #         logger.debug("Training model: trial {}".format(i))
    #         self.checkpoint_filename = "udf={}-max_positive_samples={}-trial={}".format(self.gt_udf_name, self.max_positive_samples, i)
    #         os.makedirs(self.checkpoint_root, exist_ok=True)
    #         checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #             filename=self.checkpoint_filename,
    #             monitor="val_loss",
    #             mode="min",
    #         )
    #         callbacks=[checkpoint_callback]
    #         earlystopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    #         callbacks.append(earlystopping_callback)
    #         lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    #         callbacks.append(lr_monitor)

    #         self.mlp_model = mlp.MLP(mlp_dim_in, 2, logger, self.class_weights) # binary classification

    #         self.trainer = pl.Trainer(
    #             # deterministic=self.deterministic,
    #             max_epochs=50,
    #             devices=1,
    #             accelerator="auto",
    #             enable_progress_bar=True,
    #             enable_checkpointing=True,
    #             enable_model_summary=False,
    #             # logger=pl_logger,
    #             default_root_dir=self.checkpoint_root,
    #             callbacks=callbacks,
    #             # check_val_every_n_epoch=5,
    #             # log_every_n_steps=min(50, len(dataset)-1),
    #             log_every_n_steps=1
    #         )

    #         self.trainer.fit(
    #             self.mlp_model,
    #             train_dataloaders=self.train_loader,
    #             val_dataloaders=self.val_loader
    #         )

    #         # retrieve the best checkpoint after training
    #         current_model_score = checkpoint_callback.best_model_score
    #         logger.debug(f"current_model_score: {current_model_score}, best_model_score: {min(best_model_score, current_model_score)}")
    #         if current_model_score < best_model_score:
    #             best_model_score = current_model_score
    #             best_ckpt = checkpoint_callback.best_model_path
    #     logger.debug("Best model checkpoint: {}".format(best_ckpt))

    #     best_mlp_model = mlp.MLP.load_from_checkpoint(best_ckpt)

    #     trainer = pl.Trainer(devices=1, accelerator='auto')
    #     val_result = trainer.validate(best_mlp_model, dataloaders=self.val_loader)
    #     logger.info(f"validation result: {val_result}")

    #     class_counts = [sum(data["llm_label"] == i for data in self.labeled_data['test']) for i in range(2)]
    #     logger.info(f"test data: #pos={class_counts[1]}, #neg={class_counts[0]}")
    #     test_result = trainer.test(best_mlp_model, dataloaders=self.test_loader)
    #     logger.info(f"test result: {test_result}")

    #     return best_ckpt

    def construct_balanced_data(self, n_obj, n_train_distill):
        if self.dataset == "charades":
            return self._construct_balanced_data_with_images_charades(n_obj, n_train_distill)
        elif self.dataset == "cityflow":
            return self._construct_balanced_data_with_images_cityflow(n_obj, n_train_distill)
        else:
            raise NotImplementedError

    def _construct_balanced_data_with_images_charades(self, n_obj, n_train_distill):
        # Construct training data and test data
        # dataframe consists of columns: img, o1 [, o2]
        df = self._construct_balanced_data_without_images_charades(n_obj, n_train_distill)
        df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
        return df

    def _construct_balanced_data_without_images_charades(self, n_obj, n_train_distill):
        # Construct training data and test data
        # Only consider person-object relationships
        # vid, fid, o1_oid, o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oid, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, o1_o2_gt_rnames, height, width
        train_pos_sql = """
            SELECT
                o1.vid AS vid, o1.fid AS fid,
                o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2,
                ARRAY[r.rname] as o1_o2_gt_rnames
            FROM {}_objects o1, {}_objects o2, {}_relationships r
            WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid
                AND o1.vid = r.vid AND o1.fid = r.fid AND o1.oid = r.oid1 AND o2.oid = r.oid2 AND r.rname = '{}'
                AND o1.oid = 0 AND o1.vid < 3800
            ORDER BY random()
            LIMIT {}
        """.format(
            self.dataset, self.dataset, self.dataset, self.gt_udf_name, n_train_distill // 2
        )
        df_train_pos = self.conn.execute(train_pos_sql).df()
        num_train_pos = len(df_train_pos)
        logger.debug("num_train_pos: {}".format(num_train_pos))

        train_neg_sql = """
            SELECT
                o1.vid AS vid, o1.fid AS fid,
                o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2,
                ARRAY[]::varchar[] as o1_o2_gt_rnames
            FROM {}_objects o1, {}_objects o2
            WHERE o1.vid = o2.vid AND o1.fid = o2.fid AND o1.oid != o2.oid AND o1.oid = 0 AND o1.vid < 3800
                AND NOT EXISTS (
                    SELECT 1
                    FROM {}_relationships r
                    WHERE r.vid = o1.vid AND r.fid = o1.fid AND r.oid1 = o1.oid
                        AND r.oid2 = o2.oid AND r.rname = '{}'
                )
            ORDER BY random()
            LIMIT {}
        """.format(
            self.dataset, self.dataset, self.dataset, self.gt_udf_name, n_train_distill // 2
        )
        df_train_neg = self.conn.execute(train_neg_sql).df()
        num_train_neg = len(df_train_neg)
        logger.debug("num_train_neg: {}".format(num_train_neg))

        df_train = pd.concat([df_train_pos, df_train_neg], ignore_index=True)
        df_train = df_train.reset_index(drop=True)

        return df_train

    def _construct_balanced_data_with_images_cityflow(self, n_obj, n_train_distill):
        df = self._construct_balanced_data_without_images_cityflow(n_obj, n_train_distill)
        df["img"] = list(tqdm(self.executor.map(self.frame_processing_for_program, df["vid"], df["fid"]), total=len(df), file=sys.stdout, desc="Processing frames"))
        return df

    def _construct_balanced_data_without_images_cityflow(self, n_obj, n_train_distill):
        # train: 0-659, val: 660-823, test: 824-1647
        train_pos_sql = """
            SELECT
                o1.vid AS vid, o1.fid AS fid,
                o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                ARRAY[a1.aname] AS o1_gt_anames
            FROM {}_objects o1, {}_attributes a1
            WHERE o1.vid = a1.vid AND o1.fid = a1.fid AND o1.oid = a1.oid AND a1.aname = '{}' AND o1.vid < 660
            ORDER BY random()
            LIMIT {}
        """.format(
            self.dataset, self.dataset, self.gt_udf_name, n_train_distill // 2
        )
        df_train_pos = self.conn.execute(train_pos_sql).df()
        num_train_pos = len(df_train_pos)
        logger.debug("num_train_pos: {}".format(num_train_pos))

        train_neg_sql = """
            SELECT
                o1.vid AS vid, o1.fid AS fid,
                o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
                ARRAY[]::varchar[] as o1_gt_anames
            FROM {}_objects o1
            WHERE o1.vid < 660
                AND NOT EXISTS (
                    SELECT 1
                    FROM {}_attributes a1
                    WHERE a1.vid = o1.vid AND a1.fid = o1.fid AND a1.oid = o1.oid AND a1.aname = '{}'
                )
            ORDER BY random()
            LIMIT {}
        """.format(
            self.dataset, self.dataset, self.gt_udf_name, n_train_distill // 2
        )
        df_train_neg = self.conn.execute(train_neg_sql).df()
        num_train_neg = len(df_train_neg)
        logger.debug("num_train_neg: {}".format(num_train_neg))

        df_train = pd.concat([df_train_pos, df_train_neg], ignore_index=True)
        df_train = df_train.reset_index(drop=True)

        return df_train

async def main():
    # python evaluate_model_udf.py --dataset "charades" --udf_name "holding" --llm_method "gpt4v" --n_train_distill 50 --balanced --run_id 0 --is_async --openai_model_name "gpt-4o"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["charades", "cityflow"], help="dataset name")
    parser.add_argument("--udf_name", type=str, help="udf_name")
    parser.add_argument("--llm_method", type=str, choices=["gpt4v", "llava", "user"], help="LLM method")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data")
    parser.add_argument("--balanced", action="store_true", help="use balanced dataset")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--is_async", action="store_true", help="use async for distilled-model UDF labeling")
    parser.add_argument("--openai_model_name", type=str, help="OpenAI model name")

    args = parser.parse_args()
    dataset = args.dataset
    udf_name = args.udf_name
    llm_method = args.llm_method
    n_train_distill = args.n_train_distill
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data
    balanced = args.balanced
    run_id = args.run_id
    is_async = args.is_async
    openai_model_name = args.openai_model_name

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
            ["touching(o0, o1)", "Whether o0 is touching o1.", "touching"],
            ["leaning_on(o0, o1)", "Whether o0 is leaning on o1.", "leaning_on"],
            ["wearing(o0, o1)", "Whether o0 is wearing o1.", "wearing"],
            ["drinking_from(o0, o1)", "Whether o0 is drinking from o1.", "drinking_from"],
            ["lying_on(o0, o1)", "Whether o0 is lying on o1.", "lying_on"],
            ["writing_on(o0, o1)", "Whether o0 is writing on o1.", "writing_on"],
            ["twisting(o0, o1)", "Whether o0 is twisting o1.", "twisting"],
            ["above(o0, o1)", "Whether o0 is above o1.", "above"],
            ["in_front_of(o0, o1)", "Whether o0 is in front of o1.", "in_front_of"],
            ["beneath(o0, o1)", "Whether o0 is beneath o1.", "beneath"],
            ["behind(o0, o1)", "Whether o0 is behind o1.", "behind"],
            ["in(o0, o1)", "Whether o0 is in o1.", "in"],
        ]
        udf_description = [t for t in test_inputs if t[2] == udf_name][0][1]
        # udf_names = ["holding", "sitting_on", "standing_on", "covered_by", "carrying", "eating", "wiping", "have_it_on_the_back", "touching", "leaning_on", "wearing", "drinking_from", "lying_on", "writing_on", "twisting", "above", "in_front_of", "beneath", "behind", "in"]
        n_obj = 2
    elif dataset == "cityflow":
        test_inputs = [
            ["suv(o0)", "Whether the type of car o0 is an SUV.", "suv"],
            ["white(o0)", "Whether the color of car o0 is white.", "white"],
            ["grey(o0)", "Whether the color of car o0 is grey.", "grey"],
            ["van(o0)", "Whether the type of car o0 is a van.", "van"],
            ["sedan(o0)", "Whether the type of car o0 is a sedan.", "sedan"],
            ["black(o0)", "Whether the color of car o0 is black.", "black"],
            ["red(o0)", "Whether the color of car o0 is red.", "red"],
            ["blue(o0)", "Whether the color of car o0 is blue.", "blue"],
            ["pickup_truck(o0)", "Whether the type of car o0 is a pickup truck.", "pickup_truck"],
        ]
        udf_description = [t for t in test_inputs if t[2] == udf_name][0][1]
        # udf_names = ["suv", "white", "grey", "van", "sedan", "black", "red", "blue", "pickup_truck"]
        n_obj = 1

    """
    Set up logging
    """
    # Create a file handler that logs even debug messages
    log_dir = os.path.join(
        config["log_dir"],
        "labeling_quality",
        dataset,
        f"balanced={balanced}",
    )
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"udf_name={udf_name}-n_train_distill={n_train_distill}-llm_method={llm_method}-run_id={run_id}.log"), mode="w")
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
    up = LabelingQuality(
            config,
            object_domain,
            relationship_domain,
            attribute_domain,
            dataset,
            num_workers,
            n_train_distill,
            llm_method,
            run_id,
            is_async,
            openai_model_name,
            udf_name
        )
    if balanced:
        await up._label_data_balanced(udf_name, udf_description, n_obj)
        # best_ckpt = up._distill_model_balanced(udf_name, n_obj)
        # logger.info(f"Best model checkpoint: {best_ckpt}")

        # Move best_ckpt to the model directory
        # best_ckpt_filename = os.path.basename(best_ckpt)
        # best_ckpt_dest_dir = os.path.join(config["data_dir"], "trained_udfs", dataset)
        # os.makedirs(best_ckpt_dest_dir, exist_ok=True)
        # best_ckpt_dest = os.path.join(best_ckpt_dest_dir, "udf={}-max_positive_samples={}.ckpt".format(udf_name, max_positive_samples))
        # shutil.move(best_ckpt, best_ckpt_dest)
        # logger.info(f"Moved {best_ckpt} to {best_ckpt_dest}")
    else:
        raise NotImplementedError
        # up._distill_model(udf_name)

if __name__ == "__main__":
    asyncio.run(main())