#!/bin/bash

# declare -a UDFArray=("shape_cylinder" "color_brown" "material_metal" "color_purple")
# declare -a UDFArray=("material_rubber" "color_cyan" "shape_cube" "shape_sphere" "near" "far" "rightof" "behind" "location_right" "location_bottom")
declare -a UDFArray=("shape_cylinder")

for udf in "${UDFArray[@]}"; do
    for run in {0..4}; do
        sbatch exp-compare_model_udf.sbatch $run $udf
        # sbatch exp-compare_code_udf.sbatch $run $udf
    done
done