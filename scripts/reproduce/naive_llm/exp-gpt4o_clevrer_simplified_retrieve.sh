#!/bin/bash

declare -a TaskArray=("simplified_3_new_udfs_labels")

for query_filename in "${TaskArray[@]}"; do
    for run in 0; do
        for query_id in {0..2}; do
            python \
                $PROJECT_ROOT/experiments/gpt4o_clevrer_simplified.py \
                --run_id $run \
                --query_id $query_id \
                --query_filename "$query_filename" \
                --openai_model_name "gpt-4o" \
                --stage "retrieve"
        done
    done
done