#!/bin/bash

# charades
declare -a UDFArray=("holding" "sitting_on" "standing_on" "covered_by" "carrying" "eating" "wiping" "have_it_on_the_back" "touching" "leaning_on" "wearing" "drinking_from" "lying_on" "writing_on" "twisting" "above" "in_front_of" "beneath" "behind" "in")

for run_id in 0 1 2; do
# for run_id in 0; do
    for udf_name in "${UDFArray[@]}"; do
        sbatch exp-model_udf.sbatch "charades" $udf_name $run_id
    done
done

# cityflow
declare -a UDFArray=("suv" "white" "grey" "van" "sedan" "black" "red" "blue" "pickup_truck")
# declare -a UDFArray=("pickup_truck")

for run_id in 0 1 2; do
# for run_id in 0; do
    for udf_name in "${UDFArray[@]}"; do
        sbatch exp-model_udf.sbatch "cityflow" $udf_name $run_id
    done
done
