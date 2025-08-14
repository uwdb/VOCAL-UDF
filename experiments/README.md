# Reproduce Experiments

## End-to-end evaluation (Figure 5, 7)

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
2. Visualize results: `experiments/analysis/end_to_end.ipynb`

## Natural language to DSL correctness (Table 4)
1. Run experiments
```bash
./scripts/reproduce/nl_to_dsl/exp-nl_to_dsl_cityflow_charades.sh
./scripts/reproduce/nl_to_dsl/exp-nl_to_dsl_clevrer.sh
```
2. Visualize results: `experiments/analysis/mircobenchmark.ipynb`, "NL TO DSL" section

## Comparing with direct LLM methods (Figure 6)
This experiment uses OpenAI’s Batch API that offers 50% lower cost.

1. Submit batch jobs
```bash
./scripts/reproduce/naive_llm/exp-gpt4o_clevrer_simplified_submit.sh
```

2. Retrieve results
```bash
./scripts/reproduce/naive_llm/exp-gpt4o_clevrer_simplified_retrieve.sh
```

3. Visualize results


## UDF proposal (Section 6.2)
## UDF generation (Section 6.3)
## UDF selection (Section 6.4)
