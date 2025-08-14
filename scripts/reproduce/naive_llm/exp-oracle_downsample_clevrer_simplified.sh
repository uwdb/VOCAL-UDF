#!/bin/bash

for sampling_rate in 1 2 4 8; do
    for query_id in {0..9}; do
        python \
            $PROJECT_ROOT/experiments/oracle_downsample_clevrer_simplified.py \
            --query_id $query_id \
            --query_filename "simplified_3_new_udfs_labels" \
            --sampling_rate $sampling_rate
    done
done