#!/bin/bash

# CityFlow
declare -a QueryClassArray=("unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737")
# declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2")

for query_class_name in "${QueryClassArray[@]}"; do
    for budget in 50; do
        for num_interpretations in 10; do
            # for run in 0 1 2 3 4; do
            for run in 0 1 2; do
                for query_id in {0..14}; do
                    sbatch exp-nl_to_dsl_cityflow_charades.sbatch $query_id $run "cityflow" $query_class_name $budget $num_interpretations
                done
            done
        done
    done
done


# Charades
declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2" "unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2" "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2")
# declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2")

for query_class_name in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2; do
        for budget in 50; do
            for num_interpretations in 10; do
                for run in 0 1 2; do
                    for query_id in {0..9}; do
                        sbatch exp-nl_to_dsl_cityflow_charades.sbatch $query_id $run "charades" $query_class_name $budget $num_interpretations
                    done
                done
            done
        done
    done
done
