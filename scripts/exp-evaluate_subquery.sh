#!/bin/bash

# Charades
# declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2" "unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2" "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2")
# declare -a QueryClassArray=("unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2")

# for query_class_name in "${QueryClassArray[@]}"; do
#     for num_missing_udfs in 2; do
#         for query_id in {0..9}; do
#             sbatch exp-evaluate_subquery.sbatch $num_missing_udfs $query_id "charades" $query_class_name
#         done
#     done
# done

# GQA
# declare -a QueryClassArray=("unavailable=2-npred=1-nattr_pred=1-nobj_pred=0-nvars=2-min_npos=100-max_npos=5000" "unavailable=2-npred=1-nattr_pred=2-nobj_pred=0-nvars=3-min_npos=100-max_npos=5000")
# declare -a QueryClassArray=("unavailable=2-npred=1-nattr_pred=2-nobj_pred=0-nvars=3-min_npos=100-max_npos=5000")

# for query_class_name in "${QueryClassArray[@]}"; do
#     for num_missing_udfs in 1 2; do
#         for query_id in {0..9}; do
#             sbatch exp-evaluate_subquery.sbatch $num_missing_udfs $query_id "gqa" $query_class_name
#         done
#     done
# done

# VAW
# declare -a QueryClassArray=("unavailable=2-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000" "")
declare -a QueryClassArray=("unavailable=2-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000")

for query_class_name in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 1 2; do
        for query_id in {0..9}; do
            sbatch exp-evaluate_subquery.sbatch $num_missing_udfs $query_id "vaw" $query_class_name
        done
    done
done