#!/bin/bash

declare -a QueryClassArray=("unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737")

for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2; do
        for budget in 50; do
            for num_interpretations in 10; do
                for run in 0 1 2; do
                    for query_id in {0..14}; do
                        python \
                            $PROJECT_ROOT/experiments/run_udf_selection_random_count.py \
                            --num_missing_udfs $num_missing_udfs \
                            --run_id $run \
                            --query_id $query_id \
                            --dataset "cityflow" \
                            --query_filename "$query_filename" \
                            --budget $budget \
                            --num_interpretations $num_interpretations \
                            --allow_kwargs_in_udf \
                            --num_parameter_search 5 \
                            --generate \
                            --num_workers 8 \
                            --save_labeled_data \
                            --n_train_distill 500 \
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
