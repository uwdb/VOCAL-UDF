#!/bin/bash -l
#SBATCH --job-name=exp-visprog
#SBATCH --partition=ckpt-all
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=4:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

num_missing_udfs=$1
query_id=$2
run=$3
dataset=$4
query_class_name=$5

echo "num_missing_udfs: $num_missing_udfs"
echo "query_id: $query_id"
echo "run: $run"
echo "dataset: $dataset"
echo "query_class_name: $query_class_name"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/visprog_vaw.py \
        --use_precomputed \
        --num_missing_udfs $num_missing_udfs \
        --dataset "$dataset" \
        --query_class_name "$query_class_name" \
        --run_id $run \
        --query_id $query_id \
        --llm_model "gpt-4-turbo-2024-04-09"

# python visprog_vaw.py --use_precomputed --num_missing_udfs 1 --dataset "vaw" --query_class_name "unavailable_pred=1-unavailable_attr_pred=1-npred=2-nattr_pred=1-nvars=3-min_npos=3000-max_npos=20000" --run_id 0 --query_id 0 --llm_model "gpt-4-turbo-2024-04-09"