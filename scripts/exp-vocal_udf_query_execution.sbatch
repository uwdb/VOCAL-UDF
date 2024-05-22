#!/bin/bash -l
#SBATCH --job-name=query_exec
#SBATCH --partition=ckpt
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


run=$1
query_id=$2
dataset=$3
budget=$4
num_interpretations=$5
num_missing_udfs=$6

echo "run: $run"
echo "query_id: $query_id"
echo "dataset: $dataset"
echo "budget: $budget"
echo "num_interpretations: $num_interpretations"
echo "num_missing_udfs: $num_missing_udfs"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/run_query_executor.py \
        --num_missing_udfs $num_missing_udfs \
        --run_id $run \
        --query_id $query_id \
        --dataset "$dataset" \
        --budget $budget \
        --num_interpretations $num_interpretations \
        --allow_kwargs_in_udf \
        --num_parameter_search 10 \
        --program_with_pixels \
        --cpus 8 \
        --n_train_distill 100 \
        --selection_strategy "both" \
        --selection_labels "user" \
        --pred_batch_size 4096 \
        --dali_batch_size 1

# python run_query_executor.py --num_missing_udfs 3 --query_id 3 --run_id 0 --dataset "clevrer" --budget 20 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 10 --program_with_pixels --cpus 8 --n_train_distill 100 --selection_strategy "both" --selection_labels "user"