#!/bin/bash

for n_train_distill in 500; do
    for udf_idx in {0..9}; do
        sbatch test_distill_model.sbatch $udf_idx $n_train_distill "charades"
    done
done
