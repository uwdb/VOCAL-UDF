#!/bin/bash

# Charades
# declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2" "unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2" "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2")
declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2")

for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2; do
        for run_id in 0; do
        # for run_id in 0 1 2; do
            for query_id in {0..2}; do
            # for query_id in {0..9}; do
                python \
                    $PROJECT_ROOT/experiments/evaluate_equivocal.py \
                    --num_missing_udfs $num_missing_udfs \
                    --query_id $query_id \
                    --run_id $run_id \
                    --dataset "charades" \
                    --query_filename "$query_filename" \
                    --num_workers 4 \
                    --pred_batch_size 4096 \
                    --dali_batch_size 1 \
                    --openai_model_name "gpt-4o"
            done
        done
    done
done

# Cityflow
# declare -a QueryClassArray=("unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737")

# for query_filename in "${QueryClassArray[@]}"; do
#     for num_missing_udfs in 0 1 2; do
#         for run_id in 0 1 2; do
#             for query_id in {0..14}; do
#                 python \
#                     $PROJECT_ROOT/experiments/evaluate_equivocal.py \
#                     --num_missing_udfs $num_missing_udfs \
#                     --query_id $query_id \
#                     --run_id $run_id \
#                     --dataset "cityflow" \
#                     --query_filename "$query_filename" \
#                     --num_workers 4 \
#                     --pred_batch_size 4096 \
#                     --dali_batch_size 1 \
#                     --openai_model_name "gpt-4o"
#             done
#         done
#     done
# done

# # Clevrer
# declare -a QueryClassArray=("3_new_udfs_labels")

# for query_filename in "${QueryClassArray[@]}"; do
#     for num_missing_udfs in 0 1 2 3; do
#         for run_id in 0 1 2; do
#             for query_id in {0..29}; do
#                 python \
#                     $PROJECT_ROOT/experiments/evaluate_equivocal.py \
#                     --num_missing_udfs $num_missing_udfs \
#                     --query_id $query_id \
#                     --run_id $run_id \
#                     --dataset "clevrer" \
#                     --query_filename "$query_filename" \
#                     --num_workers 4 \
#                     --pred_batch_size 4096 \
#                     --dali_batch_size 1 \
#                     --openai_model_name "gpt-4o"
#             done
#         done
#     done
# done