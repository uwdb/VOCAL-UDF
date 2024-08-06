#!/bin/bash

# Clevrer
declare -a QueryClassArray=("3_new_udfs_labels")

for query_class_name in "${QueryClassArray[@]}"; do
    for budget in 20; do
        for num_interpretations in 10; do
            for run in 0 1 2; do
                for query_id in {0..29}; do
                    sbatch exp-llm_decides_udf_type_clevrer.sbatch $query_id $run "clevrer" $query_class_name $budget $num_interpretations
                done
            done
        done
    done
done
