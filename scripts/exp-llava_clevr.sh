#!/bin/bash

declare -a TaskArray=("3_new_udfs_labels")
# declare -a TaskArray=("0_new_udfs_labels" "1_new_udfs_labels" "2_new_udfs_labels" "3_new_udfs_labels")

for task_name in "${TaskArray[@]}"; do
    for run in {0..4}; do
        for question_id in {0..9}; do
            sbatch exp-llava_clevr.sbatch $task_name $run $question_id
        done
    done
done