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
query_filename=$4
budget=$5
num_interpretations=$6

echo "query_id: $query_id"
echo "run: $run"
echo "dataset: $dataset"
echo "query_filename: $query_filename"
echo "budget: $budget"
echo "num_interpretations: $num_interpretations"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/evaluate_llm_decides_udf_type.py \
        --run_id $run \
        --query_id $query_id \
        --dataset "$dataset" \
        --query_filename "$query_filename" \
        --budget $budget \
        --num_interpretations $num_interpretations \
        --allow_kwargs_in_udf \
        --num_parameter_search 5 \
        --n_train_distill 500 \
        --selection_strategy "both" \
        --llm_method "gpt" \
        --is_async \
        --openai_model_name "gpt-4-turbo-2024-04-09"

# python run_query_executor.py --num_missing_udfs 3 --query_id 0 --run_id 0 --dataset "charades" --budget 20 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 10 --num_workers 8 --n_train_distill 100 --selection_strategy "both"