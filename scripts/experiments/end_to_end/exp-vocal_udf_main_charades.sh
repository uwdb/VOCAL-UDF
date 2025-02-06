#!/bin/bash

declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=3-depth=2" "unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2" "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2")
# declare -a QueryClassArray=("unavailable=2-npred=4-nobj_pred=1-nvars=2-depth=2" "unavailable=2-npred=3-nobj_pred=1-nvars=2-depth=2")

for query_filename in "${QueryClassArray[@]}"; do
    for num_missing_udfs in 0 1 2; do
    # for num_missing_udfs in 2; do
        for budget in 50; do
            for num_interpretations in 10; do
                # for run in 0 1 2 3 4; do
                for run in 2; do
                    # for query_id in {5..9}; do
                    for query_id in {0..9}; do
                        sbatch exp-vocal_udf_main_charades.sbatch $query_id $run "charades" $query_filename $budget $num_interpretations $num_missing_udfs
                    done
                done
            done
        done
    done
done
