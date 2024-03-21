#!/bin/bash

# declare -a UDFArray=("shape_cylinder" "color_brown" "material_metal")
declare -a UDFArray=("color_purple")

for udf in "${UDFArray[@]}"; do
    for run in {0..4}; do
        sbatch exp-compare_model_udf.sbatch $run $udf
        sbatch exp-compare_code_udf.sbatch $run $udf
    done
done