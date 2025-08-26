#!/bin/bash

declare -a TaskArray=("simplified_3_new_udfs_labels")

for query_filename in "${TaskArray[@]}"; do
    for run in 0; do
        for query_id in {1..9}; do
            sbatch exp-gpt_clevrer_simplified.sbatch $query_filename $run $query_id
        done
    done
done