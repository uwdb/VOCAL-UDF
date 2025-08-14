#!/bin/bash

declare -a QueryClassArray=("simplified_3_new_udfs_labels")

for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2 3; do
    # for num_missing_udfs in 3; do
        for budget in 20; do
            for num_interpretations in 10; do
                # for run in 0 1 2; do
                for run in 0; do
                    for query_id in {0..9}; do
                        python \
                            $PROJECT_ROOT/experiments/run_clevrer_simplified_query.py \
                            --num_missing_udfs $num_missing_udfs \
                            --run_id $run \
                            --query_id $query_id \
                            --dataset "clevrer" \
                            --query_filename "$query_filename" \
                            --budget $budget \
                            --n_selection_samples 500 \
                            --num_interpretations $num_interpretations \
                            --allow_kwargs_in_udf \
                            --program_with_pixels \
                            --num_parameter_search 5 \
                            --num_workers 8 \
                            --n_train_distill 100 \
                            --selection_strategy "both" \
                            --pred_batch_size 4096 \
                            --dali_batch_size 1 \
                            --llm_method "gpt" \
                            --openai_model_name "gpt-4o"
                    done
                done
            done
        done
    done
done