#!/bin/bash

declare -a UdfNameArray=("far" "behind" "location_bottom")

for udf_name in "${UdfNameArray[@]}"; do
    for interpretation_id in 0 1 2; do
        for budget in 20; do
            for num_interpretations in 10; do
                for run in {0..19}; do
                    sbatch exp-intent_ambiguity_clevrer.sbatch $udf_name $run $interpretation_id $budget $num_interpretations
                done
            done
        done
    done
done