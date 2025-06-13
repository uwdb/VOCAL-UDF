"""
cli.py - Interactive terminal UI for the VOCAL-UDF pipeline
===========================================================
This CLI lets a human step into three points of the pipeline:

1. Supply the natural-language query.
2. Pick which *registered* UDFs to expose to VOCAL-UDF.
3. Provide binary labels for the UDF selection stage by looking at
   saved frame images in an external viewer and typing 1/0 back in
   the terminal.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
import logging
import os
import random
import resource
import time
from typing import Dict, Any, List

import numpy as np
from rich import print
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
import yaml

from vocaludf.query_parser import QueryParser
from vocaludf.udf_proposer import UDFProposer
from vocaludf.async_udf_generator import UDFGenerator
from vocaludf.interactive_udf_selector import InteractiveUDFSelector
from vocaludf.query_executor import QueryExecutor
from vocaludf.utils import parse_signature, get_active_domain, setup_logging, SharedResources

logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

async def process_udf(udf_signature, udf_description, shared_resources):
    logger.info(f"UDF generation for {udf_signature} started")
    ug = UDFGenerator(shared_resources, udf_signature, udf_description, None)
    udf_candidate_list, llm_positive_df, llm_negative_df = await ug.implement()
    logger.info(f"UDF generation for {udf_signature} finished")

    cost_estimation = ug.get_cost_estimation()
    execution_time = ug.get_execution_time()
    return (
        udf_signature,
        udf_description,
        udf_candidate_list,
        llm_positive_df,
        llm_negative_df,
        cost_estimation,
        execution_time,
    )


def choose_registered_udfs(udf_corpus: List[Any]) -> List[Any]:
    """Let the user pick a subset of UDFs to expose to the pipeline."""
    udf_list: List[tuple[str, str]] = [(udf_dict["signature"].split("(")[0], udf_dict["description"]) for udf_dict in udf_corpus if udf_dict["signature"].split("(")[0] != "object"]

    table = Table(title="Registered UDFs")
    table.add_column("#", justify="right", style="bold")
    table.add_column("UDF Name")
    table.add_column("UDF Description", style="dim")
    for i, (sig, desc) in enumerate(udf_list):
        table.add_row(str(i), sig, desc)
    print(table)

    raw = Prompt.ask("[bold cyan]Indices of UDFs to include (comma-separated, object UDFs are always included)[/bold cyan]",)
    try:
        chosen = [int(tok) for tok in raw.replace(",", " ").split()]  # type: ignore[arg-type]
    except ValueError:
        print("[red]Invalid list - falling back to none.[/red]")
        chosen = []

    return [
        udf_dict for i, udf_dict in enumerate(udf_corpus)
        if i in chosen or udf_dict["signature"].split("(")[0] == "object"
    ]


async def main() -> None:
    ### clevrer:
    # python cli.py --dataset clevrer --allow_kwargs_in_udf --num_parameter_search 5 --budget 20 --n_selection_samples 500 --num_interpretations 10 --program_with_pixels --num_workers 8 --save_labeled_data --n_train_distill 100 --selection_strategy both --llm_method gpt --is_async --openai_model_name gpt-4o
    # user query: A cyan-colored object o1 is in front of a cylinder o2, then o1 moves to be behind and close to o2.
    # registered UDFs: 1, 14, 19

    config = yaml.safe_load(
        open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
    )
    prompt_config = yaml.load(
        open(os.path.join(config["prompt_dir"], "prompt.yaml"), "r"),
        Loader=yaml.FullLoader,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--allow_kwargs_in_udf", action="store_true", help="allow kwargs in program-based UDF")
    parser.add_argument("--num_parameter_search", type=int, help="for program UDF candidate with kwargs, the number of different parameter values to explore")
    parser.add_argument("--budget", type=int, help="labeling budget per UDF for UDF selection")
    parser.add_argument("--n_selection_samples", type=int, default=500, help="number of sampled videos per selection iteration")
    parser.add_argument("--num_interpretations", type=int, help="number of semantic interpretations to generate for the program UDF")
    parser.add_argument("--program_with_pixels", action="store_true", help="allow frame pixels as inputs when generating program UDFs")
    parser.add_argument("--num_workers", type=int, default=8, help="Maximum number of tasks to execute at once")
    parser.add_argument("--save_labeled_data", action="store_true", help="save labeled data for model distillation")
    parser.add_argument("--load_labeled_data", action="store_true", help="load labeled data for model distillation")
    parser.add_argument("--n_train_distill", type=int, help="number of training samples for model distillation")
    parser.add_argument("--selection_strategy", type=str, choices=["program", "model", "llm", "both"], default="model", help="strategy for UDF selection")
    parser.add_argument("--llm_method", type=str, choices=["gpt", "llava"], default="gpt", help="LLM method for distill model annotations")
    parser.add_argument("--is_async", action="store_true", help="use async for distilled-model UDF labeling")
    parser.add_argument("--openai_model_name", type=str, help="OpenAI model name")

    args = parser.parse_args()
    dataset = args.dataset
    allow_kwargs_in_udf = args.allow_kwargs_in_udf
    num_parameter_search = args.num_parameter_search
    labeling_budget = args.budget
    n_selection_samples = args.n_selection_samples
    num_interpretations = args.num_interpretations
    program_with_pixels = args.program_with_pixels
    program_with_pretrained_models = False
    if program_with_pretrained_models:
        assert program_with_pixels, "program_with_pretrained_models requires program_with_pixels"
    # generate = args.generate
    num_workers = args.num_workers
    save_labeled_data = args.save_labeled_data
    load_labeled_data = args.load_labeled_data
    n_train_distill = args.n_train_distill
    selection_strategy = args.selection_strategy
    llm_method = args.llm_method
    is_async = args.is_async
    openai_model_name = args.openai_model_name

    random.seed(0)
    np.random.seed(0)

    # Set up logging
    base_dir = os.path.join(
        "live_query",
        dataset,
    )
    log_filename = "out.log"
    setup_logging(config, base_dir, log_filename, logger, show_debug_console=False)


    # 1. Natural-language query ------------------------------------------------
    print(Panel(f"[bold green]Welcome to VOCAL-UDF CLI![/bold green]"))
    print(Panel(f"[bold green]Expressing the query...[/bold green]"))
    query_nl = Prompt.ask("[bold cyan]Enter your query in natural language[/bold cyan]")

    # 2. Choose registered UDFs ----------------------------------------------
    print(Panel(f"[bold green]Choosing registered UDFs...[/bold green]"))
    registered_udfs_json = json.load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/registered_udfs.json", "r"))
    udf_base = registered_udfs_json[f"{dataset}_base"]
    udf_additional = [v for _, v in registered_udfs_json[dataset].items()]
    udf_corpus = udf_base + udf_additional
    registered_functions = choose_registered_udfs(udf_corpus)
    logger.debug("Registered functions: {}".format(registered_functions))
    object_domain, relationship_domain, attribute_domain = get_active_domain(config, dataset, registered_functions)
    logger.debug("Active domains: object={}, relationship={}, attribute={}".format(object_domain, relationship_domain, attribute_domain))
    available_udf_names = [parse_signature(func["signature"])[0] for func in registered_functions]
    materialized_df_names = []
    on_the_fly_udf_names = []

    cost_estimation = defaultdict(float)
    total_execution_time = defaultdict(float)
    udf_generation_execution_time = defaultdict(float)

    # 3. Parse the query -------------------------------------------------------
    logger.debug("Query parsing started")
    print(Panel(f"[bold green]Parsing the query...[/bold green]"))
    _start = time.time()
    qp = QueryParser(
        config, prompt_config, dataset, registered_functions, object_domain, 0, openai_model_name
    )
    flag = qp.parse(query_nl)
    cost_estimation['query_parser'] += qp.get_cost_estimation()
    logger.debug("Query parsing finished")
    total_execution_time['query_parsing'] += time.time() - _start
    if 'parse_no' in flag:
        print(Panel(f"[green]Query contains predicates that existing UDFs cannot resolve. Generate UDFs now...[/green]"))
        # 4. Initialize shared resources
        logger.debug("Shared resources initialization started")
        print(Panel(f"[green]Initializing shared resources... [/green]"))
        _start = time.time()
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
            None, # query_filename not used in CLI
            None, # query_id not used in CLI
            0, # run_id not used in CLI
            num_workers,
            save_labeled_data,
            load_labeled_data,
            n_train_distill,
            selection_strategy,
            allow_kwargs_in_udf,
            llm_method,
            is_async,
            openai_model_name,
            test_with_gt=False,
        )
        logger.debug("Shared resources initialization finished")
        total_execution_time['resource_init'] += time.time() - _start

        # 5. Propose missing UDFs -------------------------------------
        logger.debug("UDF proposal started")
        print(Panel(f"[green]Proposing new UDFs... [/green]"))
        _start = time.time()
        up = UDFProposer(shared_resources)
        proposed_functions = up.propose(query_nl)
        up_cost_estimation = up.get_cost_estimation()
        for key, value in up_cost_estimation.items():
            cost_estimation[key] += value
        logger.debug("UDF proposal finished")
        total_execution_time['udf_proposal'] += time.time() - _start

        # 6. generate missing UDFs, concurrently
        logger.debug("UDF generation started")
        print(Panel(f"[green]Generating new UDFs... [/green]"))
        _start = time.time()
        tasks = [
            process_udf(udf_signature, udf_description, shared_resources)
            for udf_signature, udf_description in proposed_functions.items()
        ]

        # Run tasks concurrently
        results = await asyncio.gather(*tasks)
        logger.debug("UDF generation finished")
        total_execution_time['udf_generation'] += time.time() - _start

        # 7. Interactive UDF selection / labeling --------------------------------
        logger.debug("UDF selection started")
        _start = time.time()
        for result in results:
            (
                udf_signature,
                udf_description,
                udf_candidate_list,
                llm_positive_df,
                llm_negative_df,
                cost_estimation_per_udf,
                execution_time_per_udf,
            ) = result

            # Update cost estimation from UDF generation
            for key, value in cost_estimation_per_udf.items():
                cost_estimation[key] += value

            # Update execution time from UDF generation
            for key, value in execution_time_per_udf.items():
                udf_generation_execution_time[key] += value

            logger.debug(f"UDF selection for {udf_signature} started")
            print(Panel(f"[green]Selecting UDF for {udf_signature}... [/green]"))
            us = InteractiveUDFSelector(shared_resources, llm_positive_df, llm_negative_df)
            selected_udf_candidate = us.select(None, udf_candidate_list)
            logger.debug(f"UDF selection for {udf_signature} finished")
            if selected_udf_candidate is None:
                logger.warning("No UDF candidate is selected. Skipping...")
                continue

            semantic_interpretation = selected_udf_candidate.semantic_interpretation
            function_implementation = selected_udf_candidate.function_implementation

            logger.debug(
                "Best: {}, implementation: {}".format(
                    udf_signature, function_implementation
                )
            )

            # Prepare the new UDF data
            new_udf = {
                "signature": udf_signature,
                "description": udf_description,
                "semantic_interpretation": semantic_interpretation, # New field. Unsure if we need this
                "function_implementation": function_implementation,
            }
            registered_functions.append(new_udf)

            udf_name = parse_signature(udf_signature)[0]
            if semantic_interpretation in ["model", "dummy"]:
                materialized_df_names.append(udf_name)
            else:
                on_the_fly_udf_names.append(udf_name)
        logger.debug("UDF selection finished")
        total_execution_time['udf_selection'] += time.time() - _start

        # 8. Re-parse with final UDF set ------------------------------------------
        logger.debug("Query parsing started")
        print(Panel(f"[green]Re-parsing the query with updated UDFs... [/green]"))
        _start = time.time()
        qp = QueryParser(
            config,
            prompt_config,
            dataset,
            registered_functions,
            object_domain,
            0,
            openai_model_name,
            allow_new_udfs=False,
        )
        qp.parse(query_nl)
        cost_estimation['query_parser'] += qp.get_cost_estimation()
        logger.debug("Query parsing finished")
        total_execution_time['query_parsing'] += time.time() - _start
        # Cleanup memory
        del shared_resources, up, us

    parsed_program = qp.get_parsed_program()
    parsed_dsl = qp.get_parsed_query()
    logger.info("Parsed DSL: {}".format(parsed_dsl))

    # Cleanup memory
    del qp

    # 9. Execute the query
    print(Panel(f"[green]Executing the query (over the test split)... [/green]"))
    pred_batch_size = 4096
    dali_batch_size = 1
    qe = QueryExecutor(config, dataset, object_domain, relationship_domain, attribute_domain, registered_functions, available_udf_names, materialized_df_names, on_the_fly_udf_names, program_with_pixels, num_workers, pred_batch_size, dali_batch_size)
    output_vids = qe.run(parsed_program, y_true=None, debug=False)
    logger.info("Matching vids: {}".format(output_vids))

    logger.debug("Total execution time: {}".format(total_execution_time))
    logger.debug("UDF generation execution time: {}".format(udf_generation_execution_time))
    logger.debug("Cost estimation: {}".format(cost_estimation))
    logger.debug("Peak memory usuage (in GB): {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0))


if __name__ == "__main__":
    asyncio.run(main())
