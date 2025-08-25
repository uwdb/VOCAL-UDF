#!/bin/bash

declare -a QueryClassArray=("3_new_udfs_labels")

for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 3; do
        for budget in 20; do
            for num_interpretations in 10; do
                for run in 0 1 2; do
                    for query_id in {0..29}; do
                        python \
                            $PROJECT_ROOT/experiments/run_udf_selection_random.py \
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
                            --generate \
                            --num_workers 8 \
                            --save_labeled_data \
                            --n_train_distill 100 \
                            --selection_strategy "both" \
                            --llm_method "gpt" \
                            --is_async \
                            --openai_model_name "gpt-4o"
                    done
                done
            done
        done
    done
done