#!/bin/bash

declare -a QueryClassArray=("unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737")

for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2; do
        for budget in 50; do
            for num_interpretations in 10; do
                for run in 0 1 2; do
                    for query_id in {0..14}; do
                        python \
                            $PROJECT_ROOT/experiments/run_query_executor.py \
                            --num_missing_udfs $num_missing_udfs \
                            --run_id $run \
                            --query_id $query_id \
                            --dataset "cityflow" \
                            --query_filename "$query_filename" \
                            --budget $budget \
                            --num_interpretations $num_interpretations \
                            --allow_kwargs_in_udf \
                            --num_parameter_search 5 \
                            --num_workers 8 \
                            --n_train_distill 500 \
                            --selection_strategy "both" \
                            --pred_batch_size 4096 \
                            --dali_batch_size 1 \
                            --llm_method "gpt" \
                            --udf_selection_mode "no_dummy"
                    done
                done
            done
        done
    done
done
