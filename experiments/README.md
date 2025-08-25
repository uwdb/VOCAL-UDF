# Reproduce Experiments

- [x] Figure 5, 7
- [x] Table 4
- [ ] Figure 6(a)
- [ ] Figure 6(b)
- [x] Table 5

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


## UDF proposal (Table 5)
1. Visualize results: `experiments/analysis/mircobenchmark.ipynb`, "Proposing UDFs" section

## UDF generation (Table 6, Figure 8, Table 7)

## UDF selection (Table 8, Table 9, Figure 9, Figure 10)
