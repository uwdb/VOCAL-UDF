#!/bin/bash

# declare -a QueryClassArray=("unavailable=2-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000" "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000")
declare -a QueryClassArray=("unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000")

for query_class_name in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 1 2; do
        for budget in 50; do
            for num_interpretations in 10; do
                # for run in 0 1 2 3 4; do
                for run in 1; do
                    for query_id in {0..19}; do
                        sbatch exp-vocal_udf_query_execution_vaw.sbatch $query_id $run "vaw" $query_class_name $budget $num_interpretations $num_missing_udfs
                    done
                done
            done
        done
    done
done