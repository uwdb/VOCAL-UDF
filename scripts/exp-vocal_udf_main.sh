#!/bin/bash

for budget in 10; do
    for num_interpretations in 10; do
        # for run in 0 1 2 3 4; do
        for run in 0 1 2; do
            for query_id in {0..9}; do
                sbatch exp-vocal_udf_main.sbatch $run $query_id "clevrer" $budget $num_interpretations
            done
        done
    done
done