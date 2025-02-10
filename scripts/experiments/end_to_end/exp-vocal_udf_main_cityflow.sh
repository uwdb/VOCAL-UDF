#!/bin/bash

declare -a QueryClassArray=("unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737")
# declare -a QueryClassArray=("unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737")

for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2; do
    # for num_missing_udfs in 2; do
        for budget in 50; do
            for num_interpretations in 10; do
                # for run in 0 1 2 3 4; do
                for run in 2; do
                    # for query_id in {0..14}; do
                    for query_id in {0..14}; do
                        sbatch exp-vocal_udf_main_cityflow.sbatch $query_id $run "cityflow" $query_filename $budget $num_interpretations $num_missing_udfs
                    done
                done
            done
        done
    done
done
