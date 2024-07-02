import autogen
import yaml
import random
import json
import os
from vocaludf.utils import parse_signature, StreamToLogger, exception_hook, get_active_domain
import logging
import argparse
from sentence_transformers import SentenceTransformer, util
import numpy as np
from vocaludf.query_parser import QueryParser
from vocaludf.udf_proposer import UDFProposer
from vocaludf.query_executor import QueryExecutor
import duckdb
import sys
import resource
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

# logging.basicConfig()
logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s GB'''%(point, usage/1024.0/1024.0 )

class MLP(pl.LightningModule):
    def __init__(self, in_features, out_features, logger, lr, weight=None):
        super().__init__()
        self.my_logger = logger
        self.weight = weight
        self.lr = lr
        # self.lr = config['lr']
        # self.n_layers = config['n_layers']
        # self.hidden_features = config['hidden_features']
        # self.model = nn.Sequential()
        # for i in range(self.n_layers):
        #     if i == 0:
        #         self.model.add_module('input', nn.Linear(in_features, self.hidden_features))
        #     else:
        #         self.model.add_module(f'hidden_{i}', nn.Linear(self.hidden_features, self.hidden_features))
        #     self.model.add_module(f'nonlinear_{i}', nn.ReLU())
        # self.model.add_module('output', nn.Linear(self.hidden_features, out_features))
        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
            )
        # self.hidden = nn.Linear(in_features, 128)
        # self.nonlinear = nn.ReLU()
        # self.l1 = nn.Linear(hidden_features, out_features)
        self.softmax = nn.Softmax(dim=1)

        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()
        self.test_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_f1 = torchmetrics.classification.BinaryF1Score()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x # x are the model's logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, self.weight)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.my_logger.info(f'train_loss: {self.trainer.callback_metrics["train_loss"]}')

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, self.weight)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        y_pred_probs = self.softmax(logits)
        y_pred = torch.argmax(y_pred_probs, axis=1)
        self.val_acc.update(y_pred, y)
        self.val_f1.update(y_pred, y)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        self.log('val_acc', val_acc)
        self.log('val_f1', val_f1)
        self.val_acc.reset()
        self.val_f1.reset()
        self.my_logger.info(f'val_loss: {self.trainer.callback_metrics["val_loss"]}')
        self.my_logger.info(f'val_acc: {val_acc}, val_f1: {val_f1}')

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_pred_probs = self.softmax(logits)
        y_pred = torch.argmax(y_pred_probs, axis=1)
        self.test_acc.update(y_pred, y)
        self.test_f1.update(y_pred, y)

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute()
        test_f1 = self.test_f1.compute()
        self.log('test_acc', test_acc)
        self.log('test_f1', test_f1)
        self.test_acc.reset()
        self.test_f1.reset()
        self.my_logger.info(f'test_acc: {test_acc}, test_f1: {test_f1}')

    def predict_step(self, batch, batch_idx):
        row, x = batch
        logits = self(x)
        y_pred_probs = self.softmax(logits)
        y_pred = torch.argmax(y_pred_probs, axis=1)
        return row, y_pred

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=self.lr)
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class TestVawModelTraining(UDFProposer):
    def __init__(self,
        config,
        prompt_config,
        registered_functions,
        object_domain,
        relationship_domain,
        attribute_domain,
        dataset,
        labeling_budget,
        num_interpretations,
        num_parameter_search,
        program_with_pixels,
        program_with_pretrained_models,
        query_id,
        run_id,
        num_workers,
        save_labeled_data,
        load_labeled_data,
        n_train_distill,
        selection_strategy,
        selection_labels,
        allow_kwargs_in_udf,
        llm_method,
        lr):
        super().__init__(
            config,
            prompt_config,
            registered_functions,
            object_domain,
            relationship_domain,
            attribute_domain,
            dataset,
            labeling_budget,
            num_interpretations,
            num_parameter_search,
            program_with_pixels,
            program_with_pretrained_models,
            query_id,
            run_id,
            num_workers,
            save_labeled_data,
            load_labeled_data,
            n_train_distill,
            selection_strategy,
            selection_labels,
            allow_kwargs_in_udf,
            llm_method,
        )
        self.lr = lr
        logger.debug("lr: {}".format(self.lr))

    def train(self, active_learning_round=-1):
        # logger.debug("mlp_config: {}".format(mlp_config))
        mlp_dim_in = self.dim_in if self.n_obj == 1 else self.dim_in * 3
        logger.debug("mlp_dim_in: {}".format(mlp_dim_in)) # should be 512 for clip-vit-base-patch32
        self.checkpoint_root = os.path.join(self.config["model_dir"], "model_udf", self.dataset, f"{self.llm_method}_{self.mlp_method}", self.udf_class)
        if active_learning_round >= 0:
            self.checkpoint_root = os.path.join(self.checkpoint_root, f"active_learning_round_{active_learning_round}")
        self.checkpoint_filename = "udf-{}_run-{}_ntrain-{}".format(self.udf_class, self.run_id, self.n_train_distill)
        os.makedirs(self.checkpoint_root, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=self.checkpoint_filename,
            monitor="val_loss",
            mode="min",
        )
        callbacks=[checkpoint_callback]
        earlystopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
        callbacks.append(earlystopping_callback)
        # TODO: fine-tuning the learning rate
        learningrate_callback = pl.callbacks.LearningRateFinder()
        callbacks.append(learningrate_callback)

        logger.debug("lr: {}".format(self.lr))
        self.mlp_model = MLP(mlp_dim_in, 2, logger, self.lr, self.class_weights) # binary classification

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
        best_ckpt = checkpoint_callback.best_model_path
        logger.debug("Best model checkpoint: {}".format(best_ckpt))
        # best_mlp_model = mlp.MLP.load_from_checkpoint(best_ckpt)
        return best_ckpt

if __name__ == "__main__":
    # charades: python main.py --num_missing_udfs 3 --query_id 0 --run_id 0 --dataset "charades" --budget 20 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 10 --program_with_pixels --generate --cpus 8 --save_labeled_data --n_train_distill 100 --selection_strategy "both" --selection_labels "user" --llm_method "llava"
    # gqa: python main.py --num_missing_udfs 1 --query_id 0 --run_id 0 --dataset "gqa" --query_class_name "unavailable=2-npred=1-nattr_pred=1-nobj_pred=0-nvars=2-min_npos=100-max_npos=5000" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5 --program_with_pixels --generate --cpus 8 --save_labeled_data --n_train_distill 100 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v"
    # vaw: python main.py --num_missing_udfs 1 --query_id 0 --run_id 0 --dataset "vaw" --query_class_name "unavailable=2-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000" --budget 20 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5 --program_with_pixels --generate --cpus 8 --save_labeled_data --n_train_distill 100 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v"
    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_missing_udfs", type=int, help="number of missing UDFs")
    parser.add_argument("--query_id", type=int, help="query id")
    parser.add_argument("--run_id", type=int, help="run id")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--query_class_name", type=str, help="query class name")
    parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in UDF")
    parser.add_argument("--num_parameter_search", type=int, help="for udf candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget")
    parser.add_argument("--ask_for_gt_udf", action="store_true", help="Ask for the gt_udf name interactively if enabled")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the UDF class")
    parser.add_argument("--program_with_pixels", action="store_true", help="program with pixels")
    parser.add_argument("--program_with_pretrained_models", action="store_true", help="program with pretrained models")
    parser.add_argument('--generate', action='store_true', help="only run the UDF generation step instead of actually executing the final query.")
    parser.add_argument("--cpus", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--selection_labels", type=str, choices=["none", "user", "llm"], default="user", help="strategy for UDF selection")
    parser.add_argument("--llm_method", type=str, choices=["gpt4v", "llava"], default="gpt4v", help="LLM method for distill model annotations")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate for MLP")

    args = parser.parse_args()
    num_missing_udfs = args.num_missing_udfs
    query_id = args.query_id
    run_id = args.run_id
    dataset = args.dataset
    query_class_name = args.query_class_name
    allow_kwargs_in_udf = args.allow_kwargs_in_udf
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    ask_for_gt_udf = args.ask_for_gt_udf
    num_interpretations = args.num_interpretations
    program_with_pixels = args.program_with_pixels
    program_with_pretrained_models = args.program_with_pretrained_models
    if program_with_pretrained_models:
        assert program_with_pixels, "program_with_pretrained_models requires program_with_pixels"
    generate = args.generate
    num_workers = args.cpus
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    selection_labels = args.selection_labels
    llm_method = args.llm_method
    lr = args.lr
    # if selection_strategy != "program":
    #     assert program_with_pixels, "selection_strategy != 'program' requires program_with_pixels"

    if selection_strategy == "both":
        assert selection_labels != "none"
    elif selection_strategy == "model":
        assert selection_labels == "none"

    config_name = "ninterp={}-nparams={}-kwargs={}-pixels={}-pretrained_models={}-ntrain_distill={}-selection={}-labels={}-budget={}-llm_method={}".format(
        num_interpretations,
        num_parameter_search,
        allow_kwargs_in_udf,
        program_with_pixels,
        program_with_pretrained_models,
        n_train_distill,
        selection_strategy,
        selection_labels,
        labeling_budget,
        llm_method,
    )

    random.seed(run_id)
    np.random.seed(run_id)

    input_query_file = os.path.join(config["data_dir"], dataset, f"{query_class_name}.json")
    input_query = json.load(open(input_query_file, "r"))["questions"][query_id]
    gt_dsl = input_query["dsl"]
    user_query = input_query["question"]
    positive_videos = input_query["positive_videos"]
    if dataset in ["gqa", "vaw"]:
        conn = duckdb.connect(
            database=os.path.join(config["db_dir"], "annotations.duckdb"),
            read_only=True,
        )
        vids = conn.execute(f"SELECT DISTINCT vid FROM {dataset}_metadata ORDER BY vid ASC").df()["vid"].tolist()
        y_true = [1 if vid in positive_videos else 0 for vid in vids]
    else:
        y_true = [1 if i in positive_videos else 0 for i in range(config[dataset]["dataset_size"])]

    """
    Set up logging
    """
    base_dir = os.path.join(
        "udf_generation",
        dataset + "test",
        query_class_name,
        "num_missing_udfs={}".format(num_missing_udfs),
        "lr={}".format(lr),
        config_name,
    )
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        base_dir
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(os.path.join(log_dir, "qid={}-run={}.log".format(query_id, run_id)), mode="w")
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

    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )
    registered_udfs_json = json.load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r"))
    if "single_semantic" in query_class_name:
        registered_functions = [{
            "signature": "object(o0, name)",
            "description": "Whether o0 is an object with the given name.",
            "function_implementation": ""
        }]
    else:
        registered_functions = registered_udfs_json[f"{dataset}_base"]
        new_modules = input_query["new_modules"]
        assert num_missing_udfs >= 0 and num_missing_udfs <= 3, "num_missing_udfs must be between 0 and 3"
        for new_module in new_modules[:(len(new_modules)-num_missing_udfs)]:
            registered_functions.append(registered_udfs_json[dataset][new_module])
    logger.info("Registered functions: {}".format(registered_functions))
    object_domain, relationship_domain, attribute_domain = get_active_domain(config, dataset, registered_functions)
    logger.info("Active domains: object={}, relationship={}, attribute={}".format(object_domain, relationship_domain, attribute_domain))
    available_udf_names = [parse_signature(func["signature"])[0] for func in registered_functions]
    materialized_df_names = []
    on_the_fly_udf_names = []

    # Parse query
    qp = QueryParser(
        config, prompt_config, dataset, registered_functions, object_domain, run_id
    )
    flag = qp.parse(user_query)
    if 'parse_no' in flag:
        # Step 1: propose new UDFs
        up = TestVawModelTraining(
            config,
            prompt_config,
            registered_functions,
            object_domain,
            relationship_domain,
            attribute_domain,
            dataset,
            labeling_budget,
            num_interpretations,
            num_parameter_search,
            program_with_pixels,
            program_with_pretrained_models,
            query_id,
            run_id,
            num_workers,
            save_labeled_data,
            load_labeled_data,
            n_train_distill,
            selection_strategy,
            selection_labels,
            allow_kwargs_in_udf,
            llm_method,
            lr,
        )
        proposed_functions = up.propose(user_query)
        for udf_signature, udf_description in proposed_functions.items():
            # First, retrieve the ground truth UDF
            if ask_for_gt_udf:
                # Ask the user for gt_udf name
                if dataset == "clevrer":
                    gt_udf_name = input(
                        'Please enter gt_udf_name (options: "near", "far", "right_of", "behind", "location_right", "location_bottom", "color_brown", "color_purple", "color_cyan", "color_yellow", "shape_cylinder", "material_metal"): '
                    )
                elif dataset == "charades":
                    gt_udf_name = input(
                        'Please enter gt_udf_name (options: "looking_at", "above", "in_front_of", "on_the_side_of", "carrying", "drinking_from", "have_it_on_the_back", "leaning_on", "not_contacting", "standing_on", "twisting", "wiping", "not_looking_at", "beneath", "behind", "in", "covered_by", "eating", "holding", "lying_on", "sitting_on", "touching", "wearing", "writing_on"): '
                    )
            else:
                # HACK: Use a LM to automatically resolve the ground truth UDF
                # NOTE: Correctness is not guaranteed
                udf_name, udf_vars = parse_signature(udf_signature)
                if dataset == "clevrer":
                    if len(udf_vars) == 2:
                        gt_udf_candidates = ["near", "far", "right_of", "behind"]
                    else:
                        gt_udf_candidates = [
                            "location_right",
                            "location_bottom",
                            "color_brown",
                            "color_purple",
                            "color_cyan",
                            "color_yellow",
                            "shape_cylinder",
                            "material_metal",
                        ]
                elif dataset == "charades":
                    gt_udf_candidates = [
                        "looking_at",
                        "above",
                        "in_front_of",
                        "on_the_side_of",
                        "carrying",
                        "drinking_from",
                        "have_it_on_the_back",
                        "leaning_on",
                        "not_contacting",
                        "standing_on",
                        "twisting",
                        "wiping",
                        "not_looking_at",
                        "beneath",
                        "behind",
                        "in",
                        "inside",
                        "inside_of",
                        "covered_by",
                        "eating",
                        "holding",
                        "lying_on",
                        "sitting_on",
                        "touching",
                        "wearing",
                        "writing_on",
                    ]
                elif dataset == "gqa":
                    if len(udf_vars) == 2:
                        gt_udf_candidates = [
                            "on",
                            "near",
                            "in_front_of",
                            "next_to",
                            "above",
                            "below",
                            "on_top_of",
                            "sitting_on",
                            "carrying",
                            "to_the_left_of",
                            "to_the_right_of",
                            "wearing",
                            "of",
                            "behind",
                            "in",
                            "inside",
                            "inside_of",
                            "by",
                            "on_the_side_of",
                            "holding",
                            "walking_on",
                            "beside",
                        ]
                    else:
                        gt_udf_candidates = [
                            "black",
                            "blue",
                            "red",
                            "large",
                            "wood",
                            "tall",
                            "orange",
                            "dark",
                            "pink",
                            "clear",
                            "white",
                            "green",
                            "brown",
                            "gray",
                            "small",
                            "yellow",
                            "metal",
                            "long",
                            "silver",
                            "standing",
                        ]
                elif dataset == "vaw":
                    if len(udf_vars) == 2:
                        gt_udf_candidates = [
                            "above",
                            "beneath",
                            "to_the_left_of",
                            "to_the_right_of",
                            "in_front_of",
                            "behind",
                        ]
                    else:
                        gt_udf_candidates = [
                            "black",
                            "blue",
                            "brown",
                            "gray",
                            "small",
                            "metal",
                            "long",
                            "dark",
                            "rounded",
                            "orange",
                            "white",
                            "green",
                            "large",
                            "red",
                            "wooden",
                            "yellow",
                            "tall",
                            "silver",
                            "standing",
                            "round",
                        ]
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                gt_udf_embeddings = model.encode(gt_udf_candidates)
                implemented_udf_embedding = model.encode([udf_name])
                similarities = util.pytorch_cos_sim(
                    implemented_udf_embedding, gt_udf_embeddings
                )[0]
                gt_udf_name = gt_udf_candidates[similarities.argmax()]
                logger.debug(
                    "similarities: {}".format(
                        [
                            f"{gt_udf_candidate}: {similarity}"
                            for gt_udf_candidate, similarity in zip(
                                gt_udf_candidates, similarities
                            )
                        ]
                    )
                )
                if gt_udf_name in ["inside", "inside_of"]:
                    gt_udf_name = "in"
                logger.info(f"Selected gt_udf_name: {gt_udf_name}")
            # Step 2.a: generate semantic interpretations and implementations. Save the generated UDFs to disk
            # Step 2.b: Distilled-model UDFs
            # udf_candidate_list = up.implement(udf_signature, udf_description)
            udf_candidate_list = up.implement(udf_signature, udf_description, gt_udf_name)