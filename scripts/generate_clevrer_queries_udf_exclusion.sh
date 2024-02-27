#!/bin/bash

declare -a UDFArray=("Near" "Far" "Behind" "RightOf" "Right" "Bottom" "Color_brown" "Color_cyan" "Color_purple" "Color_yellow" "Shape_cylinder" "Material_metal")

for udf in "${UDFArray[@]}"; do
    sbatch generate_clevrer_queries_udf_exclusion.sbatch $udf
done