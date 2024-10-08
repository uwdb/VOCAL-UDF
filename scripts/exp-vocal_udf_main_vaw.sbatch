#!/bin/bash -l
#SBATCH --job-name=vaw
#SBATCH --partition=gpu-a40
#SBATCH --account=krishna
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
        /gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/main.py \
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
        --generate \
        --cpus 8 \
        --save_labeled_data \
        --n_train_distill 300 \
        --selection_strategy "both" \
        --selection_labels "user" \
        --llm_method "gpt4v"

# python main.py --num_missing_udfs 1 --query_id 1 --run_id 0 --dataset "vaw" --query_class_name "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000" --budget 50 --n_selection_samples 500 --num_interpretations 10 --allow_kwargs_in_udf --program_with_pixels --num_parameter_search 5  --generate --cpus 8 --save_labeled_data --n_train_distill 300 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v"