#!/bin/bash -l
#SBATCH --job-name=subquery
#SBATCH --partition=ckpt-all
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

num_missing_udfs=$1
query_id=$2
dataset=$3
query_class_name=$4

echo "num_missing_udfs: $num_missing_udfs"
echo "query_id: $query_id"
echo "dataset: $dataset"
echo "query_class_name: $query_class_name"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/evaluate_subquery.py \
        --num_missing_udfs $num_missing_udfs \
        --query_id $query_id \
        --dataset "$dataset" \
        --query_class_name "$query_class_name" \
        --cpus 8 \
        --pred_batch_size 4096 \
        --dali_batch_size 1