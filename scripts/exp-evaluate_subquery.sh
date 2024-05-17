#!/bin/bash

for query_id in {0..9}; do
    sbatch exp-evaluate_subquery.sbatch $query_id "clevrer"
done