#!/bin/bash

declare -a QueryClassArray=("3_new_udfs_labels")

for query_class_name in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2 3; do
        for budget in 20; do
            for num_interpretations in 10; do
                # for run in 0 1 2 3 4; do
                for run in 1 2; do
                    for query_id in {0..9}; do
                        sbatch exp-vocal_udf_main_clevrer.sbatch $query_id $run "clevrer" $query_class_name $budget $num_interpretations $num_missing_udfs
                    done
                done
            done
        done
    done
done