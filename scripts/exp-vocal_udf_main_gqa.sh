#!/bin/bash

# declare -a QueryClassArray=("unavailable=2-npred=1-nattr_pred=1-nobj_pred=0-nvars=2-min_npos=100-max_npos=5000", "unavailable=2-npred=1-nattr_pred=2-nobj_pred=0-nvars=3-min_npos=100-max_npos=5000")
declare -a QueryClassArray=("unavailable=2-npred=1-nattr_pred=2-nobj_pred=0-nvars=3-min_npos=100-max_npos=5000")

for query_class_name in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 1 2; do
        for budget in 50; do
            for num_interpretations in 10; do
                # for run in 0 1 2 3 4; do
                for run in 0; do
                    # for query_id in {5..9}; do
                    for query_id in {0..9}; do
                        sbatch exp-vocal_udf_main_gqa.sbatch $query_id $run "gqa" $query_class_name $budget $num_interpretations $num_missing_udfs
                    done
                done
            done
        done
    done
done
