#!/bin/bash -l
#SBATCH --job-name=nl_to_dsl
#SBATCH --partition=ckpt-all
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"


query_id=$1
run=$2
dataset=$3
query_class_name=$4
budget=$5
num_interpretations=$6

echo "query_id: $query_id"
echo "run: $run"
echo "dataset: $dataset"
echo "query_class_name: $query_class_name"
echo "budget: $budget"
echo "num_interpretations: $num_interpretations"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/evaluate_llm_decides_udf_type.py \
        --run_id $run \
        --query_id $query_id \
        --dataset "$dataset" \
        --query_class_name "$query_class_name" \
        --budget $budget \
        --num_interpretations $num_interpretations \
        --allow_kwargs_in_udf \
        --program_with_pixels \
        --num_parameter_search 5 \
        --n_train_distill 100 \
        --selection_strategy "both" \
        --selection_labels "user" \
        --llm_method "gpt4v" \
        --is_async \
        --openai_model_name "gpt-4-turbo-2024-04-09"