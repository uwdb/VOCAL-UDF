#!/bin/bash -l
#SBATCH --job-name=no_dummy_clevrer_query_exec
#SBATCH --partition=ckpt-all
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=100G
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
num_missing_udfs=$7

echo "query_id: $query_id"
echo "run: $run"
echo "dataset: $dataset"
echo "query_class_name: $query_class_name"
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
        --query_class_name "$query_class_name" \
        --budget $budget \
        --n_selection_samples 500 \
        --num_interpretations $num_interpretations \
        --allow_kwargs_in_udf \
        --program_with_pixels \
        --num_parameter_search 5 \
        --cpus 8 \
        --n_train_distill 100 \
        --selection_strategy "both" \
        --selection_labels "user" \
        --pred_batch_size 4096 \
        --dali_batch_size 1 \
        --llm_method "gpt4v" \
        --udf_selection_mode "no_dummy"

# python run_query_executor.py --num_missing_udfs 3 --query_id 0 --run_id 0 --dataset "clevrer" --query_class_name "3_new_udfs_labels" --budget 20 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5 --cpus 8 --n_train_distill 100 --selection_strategy "both" --selection_labels "user" --pred_batch_size 4096 --dali_batch_size 1 --llm_method "gpt4v" --random_udf_selection