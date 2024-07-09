#!/bin/bash

# charades
# declare -a UDFArray=("holding" "sitting_on" "standing_on" "covered_by" "carrying" "eating" "wiping" "have_it_on_the_back" "touching" "leaning_on" "wearing" "drinking_from" "lying_on" "writing_on" "twisting")

# for udf_name in "${UDFArray[@]}"; do
#     sbatch train_udf.sbatch "charades" $udf_name
# done

# cityflow
declare -a UDFArray=("suv" "white" "grey" "van" "sedan" "black" "red" "blue" "pickup-truck")

for udf_name in "${UDFArray[@]}"; do
    sbatch train_udf.sbatch "cityflow" $udf_name
done
