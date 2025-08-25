#!/bin/bash

# charades
declare -a UDFArray=("holding" "sitting_on" "standing_on" "covered_by" "carrying" "eating" "wiping" "have_it_on_the_back" "touching" "leaning_on" "wearing" "drinking_from" "lying_on" "writing_on" "twisting" "above" "in_front_of" "beneath" "behind" "in")

for run_id in 0 1 2; do
    for udf_name in "${UDFArray[@]}"; do
        python \
            $PROJECT_ROOT/experiments/evaluate_model_udf.py \
            --dataset "charades" \
            --udf_name $udf_name \
            --llm_method "gpt" \
            --n_train_distill 500 \
            --balanced \
            --run_id $run_id \
            --is_async \
            --openai_model_name "gpt-4o"
    done
done

# cityflow
declare -a UDFArray=("suv" "white" "grey" "van" "sedan" "black" "red" "blue" "pickup_truck")

for run_id in 0 1 2; do
    for udf_name in "${UDFArray[@]}"; do
        python \
            $PROJECT_ROOT/experiments/evaluate_model_udf.py \
            --dataset "cityflow" \
            --udf_name $udf_name \
            --llm_method "gpt" \
            --n_train_distill 500 \
            --balanced \
            --run_id $run_id \
            --is_async \
            --openai_model_name "gpt-4o"
    done
done
