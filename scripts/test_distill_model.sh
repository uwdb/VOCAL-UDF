#!/bin/bash

for udf_idx in {0..9}; do
    sbatch test_distill_model.sbatch $udf_idx
done