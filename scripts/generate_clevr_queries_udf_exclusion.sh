#!/bin/bash

# declare -a UDFArray=("Behind" "RightOf" "EqualShape" "EqualColor" "Color_brown" "Color_cyan" "Color_purple" "Color_yellow" "Shape_cylinder" "Material_metal" "Size_small")
declare -a UDFArray=("RightOf" "EqualShape" "EqualColor" "Color_purple" "Color_yellow" "Shape_cylinder" "Material_metal" "Size_small")

for udf in "${UDFArray[@]}"; do
    sbatch generate_clevr_queries_udf_exclusion.sbatch $udf
done