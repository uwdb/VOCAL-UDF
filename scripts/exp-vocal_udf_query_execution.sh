#!/bin/bash

for num_missing_udfs in 3; do
    for budget in 20; do
        for num_interpretations in 10; do
            # for run in 0 1 2 3 4; do
            for run in 0; do
                for query_id in {0..9}; do
                    sbatch exp-vocal_udf_query_execution.sbatch $run $query_id "clevrer" $budget $num_interpretations $num_missing_udfs
                done
            done
        done
    done
done