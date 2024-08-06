#!/bin/bash -l
#SBATCH --job-name=cityflow
#SBATCH --partition=gpu-a40
#SBATCH --account=krishna
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=1:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"
dataset=$1
udf_name=$2
run_id=$3

echo "dataset: $dataset"
echo "udf_name: $udf_name"
echo "run_id: $run_id"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/evaluate_model_udf.py \
        --dataset $dataset \
        --udf_name $udf_name \
        --llm_method "gpt4v" \
        --n_train_distill 500 \
        --balanced \
        --run_id $run_id