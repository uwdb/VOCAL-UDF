#!/bin/bash

for sampling_rate in 1 2 4 8; do
    for query_id in {0..9}; do
        sbatch exp-oracle_downsample_clevrer_simplified.sbatch $query_id $sampling_rate
    done
done