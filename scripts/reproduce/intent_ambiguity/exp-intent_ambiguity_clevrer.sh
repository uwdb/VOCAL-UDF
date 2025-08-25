#!/bin/bash

declare -a UdfNameArray=("far" "behind" "location_bottom")

for udf_name in "${UdfNameArray[@]}"; do
    for interpretation_id in 0 1 2; do
        for budget in 20; do
            for num_interpretations in 10; do
                for run in {0..19}; do
                    python \
                        $PROJECT_ROOT/experiments/run_intent_ambiguity_clevrer.py \
                        --run_id $run \
                        --dataset "clevrer" \
                        --udf_name "$udf_name" \
                        --interpretation_id $interpretation_id \
                        --budget $budget \
                        --n_selection_samples 500 \
                        --num_interpretations $num_interpretations \
                        --allow_kwargs_in_udf \
                        --program_with_pixels \
                        --num_parameter_search 5 \
                        --num_workers 8 \
                        --n_train_distill 100 \
                        --selection_strategy "program" \
                        --llm_method "gpt" \
                        --is_async \
                        --openai_model_name "gpt-4o"
                done
            done
        done
    done
done