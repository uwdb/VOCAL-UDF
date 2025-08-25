#!/bin/bash

# declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2" "unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2" "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2")
declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2")

for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2; do
        for budget in 50; do
            for num_interpretations in 10; do
                # for run in 0 1 2; do
                for run in 0; do
                    # for query_id in {0..9}; do
                    for query_id in {0..2}; do
                        python \
                            $PROJECT_ROOT/experiments/run_query_executor.py \
                            --num_missing_udfs $num_missing_udfs \
                            --run_id $run \
                            --query_id $query_id \
                            --dataset "charades" \
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
                            --llm_method "gpt"
                    done
                done
            done
        done
    done
done
