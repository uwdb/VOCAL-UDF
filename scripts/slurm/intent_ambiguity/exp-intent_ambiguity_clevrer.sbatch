#!/bin/bash -l
#SBATCH --job-name=intent_ambiguity
#SBATCH --partition=ckpt-all
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=4:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"


udf_name=$1
run=$2
interpretation_id=$3
budget=$4
num_interpretations=$5

echo "udf_name: $udf_name"
echo "run: $run"
echo "interpretation_id: $interpretation_id"
echo "budget: $budget"
echo "num_interpretations: $num_interpretations"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/run_intent_ambiguity_clevrer.py \
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

# python run_intent_ambiguity_clevrer.py --run_id 0 --dataset "clevrer" --udf_name "far" --interpretation_id 0 --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5 --num_workers 8 --n_train_distill 100 --selection_strategy "program" --llm_method "gpt" --is_async --openai_model_name "gpt-4o"