#!/bin/bash -l
#SBATCH --job-name=charades
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=100G
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
        /gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/main.py \
        --num_missing_udfs $num_missing_udfs \
        --run_id $run \
        --query_id $query_id \
        --dataset "$dataset" \
        --budget $budget \
        --num_interpretations $num_interpretations \
        --allow_kwargs_in_udf \
        --num_parameter_search 10 \
        --generate \
        --cpus 8 \
        --save_labeled_data \
        --n_train_distill 1000 \
        --selection_strategy "both" \
        --selection_labels "user" \
        --llm_method "llava"

# python main.py --num_missing_udfs 1 --query_id 0 --run_id 0 --dataset "charades" --budget 20 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 10  --generate --cpus 8 --save_labeled_data --n_train_distill 1000 --selection_strategy "both" --selection_labels "user" --llm_method "llava"