#!/bin/bash

for num_missing_udfs in 1 2 3; do
    for query_id in {0..9}; do
        sbatch exp-evaluate_subquery.sbatch $num_missing_udfs $query_id "clevrer"
    done
done
