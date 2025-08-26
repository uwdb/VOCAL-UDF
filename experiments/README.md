# Reproduce Experiments

Scripts to run experiments are located in the `scripts/experiments` directory. Jupyter notebooks to visualize results are located in the `experiments/analysis` directory. Modify the `project_root` variable in each Jupyter notebook as necessary.

## End-to-end evaluation (Figure 5, Figure 7)

1. Run experiments
```bash
# VisProg
./scripts/reproduce/end_to_end/exp-visprog.sh

# EQUI-VOCAL
./scripts/reproduce/end_to_end/exp-evaluate_equivocal.sh

# VOCAL-UDF
./scripts/reproduce/end_to_end/exp-vocal_udf_main_clevrer.sh
./scripts/reproduce/end_to_end/exp-vocal_udf_main_cityflow.sh
./scripts/reproduce/end_to_end/exp-vocal_udf_main_charades.sh
./scripts/reproduce/end_to_end/exp-vocal_udf_query_execution_clevrer.sh
./scripts/reproduce/end_to_end/exp-vocal_udf_query_execution_cityflow.sh
./scripts/reproduce/end_to_end/exp-vocal_udf_query_execution_charades.sh
```
2. Visualize results:
- Figure 5: `experiments/analysis/end_to_end.ipynb`
- Figure 7: `experiments/analysis/execution_time.ipynb`

## Natural language to DSL correctness (Table 4)
1. Run experiments
```bash
./scripts/reproduce/nl_to_dsl/exp-nl_to_dsl_cityflow_charades.sh
./scripts/reproduce/nl_to_dsl/exp-nl_to_dsl_clevrer.sh
```
2. Visualize results: `experiments/analysis/mircobenchmark.ipynb`, "NL TO DSL" section

## Comparing with direct LLM methods (Figure 6)
This experiment uses OpenAI’s Batch API that offers 50% lower cost and can take up to 24 hours to complete.

1. Submit batch jobs
```bash
./scripts/reproduce/naive_llm/exp-gpt4o_clevrer_simplified_submit.sh
```

2. Retrieve results
```bash
./scripts/reproduce/naive_llm/exp-gpt4o_clevrer_simplified_retrieve.sh
```

3. Run other experiments
```bash
./scripts/reproduce/naive_llm/exp-vocal_udf_clevrer_simplified.sh
./scripts/reproduce/naive_llm/exp-oracle_downsample_clevrer_simplified.sh
```

4. Visualize results: `experiments/analysis/vanilla_llm.ipynb`


## UDF proposal (Table 5)
1. Visualize results: `experiments/analysis/mircobenchmark.ipynb`, "Proposing UDFs" section

## UDF generation (Table 6, Figure 8, Table 7)
1. Run experiments
```bash
# Labeling quality (Table 7)
./scripts/reproduce/model_udf_labeling_quality/exp-model_udf.sh
# Intent ambiguity (Figure 8)
./scripts/reproduce/intent_ambiguity/exp-intent_ambiguity_baseline.sh
./scripts/reproduce/intent_ambiguity/exp-intent_ambiguity_clevrer.sh
```

2. Visualize results:
- Table 6 (left columns): `experiments/analysis/mircobenchmark.ipynb`, "Program-based UDFs" section
- Table 6 (right columns): `experiments/analysis/mircobenchmark.ipynb`, "Distilled-model UDFs" section
- Table 7: `experiments/analysis/mircobenchmark.ipynb`, "Labeling quality" section
- Figure 8: `experiments/analysis/intent_ambiguity.ipynb`

## UDF selection: selection strategy (Table 8, Table 9)
1. Run the following command to post-process the log files:
```bash
python experiments/evaluate_best_udf_type.py
```

2. Run experiments with `llm` UDF generation strategy:
```bash
./scripts/reproduce/llm_decides_udf_type/exp-llm_decides_udf_type_clevrer.sh
./scripts/reproduce/llm_decides_udf_type/exp-llm_decides_udf_type_cityflow_charades.sh
```

3. Visualize results:
- Table 8 (left columns), Table 9: `experiments/analysis/mircobenchmark.ipynb`, "Selection Strategy" section
- Table 8 (right columns): `experiments/analysis/mircobenchmark.ipynb`, "Selected UDF type distribution" section


## UDF selection: active learning (Figure 9, Figure 10)
1. Run experiments
```bash
./scripts/reproduce/ablation_udf_selection/active_count/exp-vocal_udf_selection_active_clevrer.sh
./scripts/reproduce/ablation_udf_selection/active_count/exp-vocal_udf_selection_active_cityflow.sh
./scripts/reproduce/ablation_udf_selection/active_count/exp-vocal_udf_selection_active_charades.sh

./scripts/reproduce/ablation_udf_selection/no_dummy/exp-vocal_udf_selection_no_dummy_clevrer.sh
./scripts/reproduce/ablation_udf_selection/no_dummy/exp-vocal_udf_selection_no_dummy_cityflow.sh
./scripts/reproduce/ablation_udf_selection/no_dummy/exp-vocal_udf_selection_no_dummy_charades.sh
./scripts/reproduce/ablation_udf_selection/no_dummy/exp-vocal_udf_query_execution_no_dummy_clevrer.sh
./scripts/reproduce/ablation_udf_selection/no_dummy/exp-vocal_udf_query_execution_no_dummy_cityflow.sh
./scripts/reproduce/ablation_udf_selection/no_dummy/exp-vocal_udf_query_execution_no_dummy_charades.sh

./scripts/reproduce/ablation_udf_selection/random/exp-vocal_udf_selection_random_clevrer.sh
./scripts/reproduce/ablation_udf_selection/random/exp-vocal_udf_selection_random_cityflow.sh
./scripts/reproduce/ablation_udf_selection/random/exp-vocal_udf_selection_random_charades.sh
./scripts/reproduce/ablation_udf_selection/random/exp-vocal_udf_query_execution_random_clevrer.sh
./scripts/reproduce/ablation_udf_selection/random/exp-vocal_udf_query_execution_random_cityflow.sh
./scripts/reproduce/ablation_udf_selection/random/exp-vocal_udf_query_execution_random_charades.sh

./scripts/reproduce/ablation_udf_selection/random_count/exp-vocal_udf_selection_random_clevrer.sh
./scripts/reproduce/ablation_udf_selection/random_count/exp-vocal_udf_selection_random_cityflow.sh
./scripts/reproduce/ablation_udf_selection/random_count/exp-vocal_udf_selection_random_charades.sh
```

2. Visualize results: `experiments/analysis/ablation_active_learning.ipynb`