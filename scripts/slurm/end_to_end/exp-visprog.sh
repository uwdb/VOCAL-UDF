#!/bin/bash

# clevrer
declare -a QueryClassArray=("3_new_udfs_labels")
for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2 3; do
        for run in 0 1 2; do
            # for query_id in 0; do
            for query_id in {0..29}; do
                sbatch exp-visprog.sbatch $num_missing_udfs $query_id $run "clevrer" $query_filename
            done
        done
    done
done

# charades
declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2" "unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2" "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2")
for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2; do
        for run in 0 1 2; do
            # for query_id in 0; do
            for query_id in {0..9}; do
                sbatch exp-visprog.sbatch $num_missing_udfs $query_id $run "charades" $query_filename
            done
        done
    done
done

# cityflow
declare -a QueryClassArray=("unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737")
for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2; do
        for run in 0 1 2; do
            # for query_id in 0; do
            for query_id in {0..14}; do
                sbatch exp-visprog.sbatch $num_missing_udfs $query_id $run "cityflow" $query_filename
            done
        done
    done
done