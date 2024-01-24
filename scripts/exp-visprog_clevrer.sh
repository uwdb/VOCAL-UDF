#!/bin/bash

declare -a TaskArray=("0_new_udfs_labels" "1_new_udfs_labels" "2_new_udfs_labels" "3_new_udfs_labels")
# declare -a TaskArray=("2_new_udfs_labels" "3_new_udfs_labels")

declare -a LLMArray=("gpt-3.5-turbo-instruct" "gpt-3.5-turbo-1106" "gpt-4-1106-preview")

for task_name in "${TaskArray[@]}"; do
    for llm_model in "${LLMArray[@]}"; do
        for run in {0..4}; do
            for question_id in {0..9}; do
                sbatch exp-visprog_clevrer.sbatch $task_name $run $question_id $llm_model
            done
        done
    done
done