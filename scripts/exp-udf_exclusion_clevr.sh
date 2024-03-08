#!/bin/bash

declare -a MethodArray=('nl_udf_included' 'nl_udf_excluded')

# declare -a LLMArray=("gpt-3.5-turbo-instruct" "gpt-3.5-turbo-1106" "gpt-4-1106-preview")
declare -a LLMArray=("gpt-4-1106-preview")

# declare -a UDFArray=("Behind" "RightOf" "EqualShape" "EqualColor" "Color_brown" "Color_cyan" "Color_purple" "Color_yellow" "Shape_cylinder" "Material_metal")
declare -a UDFArray=("Behind" "Color_brown" "Color_cyan")

for method in "${MethodArray[@]}"; do
    for llm_model in "${LLMArray[@]}"; do
        for udf in "${UDFArray[@]}"; do
            for run in {0..4}; do
                for question_id in {0..9}; do
                    sbatch exp-udf_exclusion_clevr.sbatch $method $run $question_id $udf $llm_model
                done
            done
        done
    done
done


for llm_model in "${LLMArray[@]}"; do
    for udf in "${UDFArray[@]}"; do
        for question_id in {0..9}; do
            sbatch exp-udf_exclusion_clevr.sbatch "dsl_udf_excluded" 0 $question_id $udf $llm_model
        done
    done
done