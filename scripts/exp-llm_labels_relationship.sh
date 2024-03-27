#!/bin/bash
declare -a UDFArray=("on" "wearing" "near" "holding" "to_the_right_of" "riding")

for udf in "${UDFArray[@]}"; do
    for run in {1..4}; do
        sbatch exp-llm_labels_relationship.sbatch $run $udf
    done
done