#!/bin/bash

declare -a QueryClassArray=("3_new_udfs_labels")

for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2 3; do
        for budget in 20; do
            for num_interpretations in 10; do
                for run in 0 1 2; do
                    for query_id in {0..29}; do
                        sbatch exp-vocal_udf_query_execution_clevrer.sbatch $query_id $run "clevrer" $query_filename $budget $num_interpretations $num_missing_udfs
                    done
                done
            done
        done
    done
done