#!/bin/bash
declare -a UDFArray=("on" "wearing" "near" "holding" "to_the_right_of" "riding")
# declare -a MethodArray=("balanced_clip_unnorm_bbox" "balanced_clip_norm_bbox" "balanced_two_clip" "balanced_three_clip" "balanced_norm_bbox_only")
# declare -a MethodArray=("balanced_clip_unnorm_bbox" "balanced_clip_norm_bbox" "balanced_two_clip" "balanced_norm_bbox_only")
declare -a MethodArray=("llava_34b_balanced_three_clip")

for udf in "${UDFArray[@]}"; do
    for run in {0..4}; do
        for method in "${MethodArray[@]}"; do
            sbatch exp-llm_labels_relationship.sbatch $run $udf $method
        done
    done
done