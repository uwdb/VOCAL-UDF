import argparse
import os
from vocaludf.async_udf_generator import UDFGenerator
from concurrent.futures import ThreadPoolExecutor
import resource
import duckdb
import logging
from openai import OpenAI, AsyncOpenAI
from collections import defaultdict
from tqdm import tqdm
import sys
from transformers import CLIPProcessor, CLIPModel
import torch
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import yaml
from vocaludf.utils import setup_logging, get_active_domain, SharedResources
import asyncio

tqdm.pandas()
client = OpenAI()

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

project_root = os.getenv("PROJECT_ROOT")

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
        udf_signature,
        test_with_gt=True,
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
        self.test_with_gt = test_with_gt

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
        if self.llm_method == "gpt":
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
    parser.add_argument("--llm_method", type=str, choices=["gpt"], help="LLM method")
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
        open(os.path.join(project_root, "configs", "config.yaml"), "r")
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

    # Set up logging
    base_dir = os.path.join(
        "labeling_quality",
        dataset,
        f"balanced={balanced}",
    )
    log_filename = f"udf_name={udf_name}-n_train_distill={n_train_distill}-llm_method={llm_method}-run_id={run_id}.log"
    setup_logging(config, base_dir, log_filename, logger)

    registered_functions = [{
        "signature": "object(o0, 'name')",
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
    else:
        raise NotImplementedError

if __name__ == "__main__":
    asyncio.run(main())