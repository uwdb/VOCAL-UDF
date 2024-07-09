#!/bin/bash

# VAW
# declare -a QueryClassArray=("unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000")

# for query_class_name in "${QueryClassArray[@]}"; do
#     for num_missing_udfs in 1 2; do
#         for run in 0; do
#             # for query_id in 0; do
#             for query_id in {0..19}; do
#                 sbatch exp-visprog.sbatch $num_missing_udfs $query_id $run "vaw" $query_class_name
#             done
#         done
#     done
# done

# clevrer
declare -a QueryClassArray=("3_new_udfs_labels")
for query_class_name in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2 3; do
        for run in 0 1 2; do
            # for query_id in 0; do
            for query_id in {0..9}; do
                sbatch exp-visprog.sbatch $num_missing_udfs $query_id $run "clevrer" $query_class_name
            done
        done
    done
done