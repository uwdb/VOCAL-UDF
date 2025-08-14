#!/bin/bash

# CityFlow
declare -a QueryClassArray=("unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737")

for query_filename in "${QueryClassArray[@]}"; do
    for budget in 50; do
        for num_interpretations in 10; do
            for run in 0 1 2; do
                for query_id in {0..14}; do
                    python \
                        $PROJECT_ROOT/experiments/evaluate_llm_decides_udf_type.py \
                        --run_id $run \
                        --query_id $query_id \
                        --dataset "cityflow" \
                        --query_filename "$query_filename" \
                        --budget $budget \
                        --num_interpretations $num_interpretations \
                        --allow_kwargs_in_udf \
                        --num_parameter_search 5 \
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


# Charades
declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2" "unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2" "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2")

for query_filename in "${QueryClassArray[@]}"; do
    for budget in 50; do
        for num_interpretations in 10; do
            for run in 0 1 2; do
                for query_id in {0..9}; do
                    python \
                        $PROJECT_ROOT/experiments/evaluate_llm_decides_udf_type.py \
                        --run_id $run \
                        --query_id $query_id \
                        --dataset "charades" \
                        --query_filename "$query_filename" \
                        --budget $budget \
                        --num_interpretations $num_interpretations \
                        --allow_kwargs_in_udf \
                        --num_parameter_search 5 \
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
