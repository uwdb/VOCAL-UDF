#!/bin/bash -l
#SBATCH --job-name=distill
#SBATCH --partition=gpu-a40
#SBATCH --account=krishna
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"
udf_idx=$1
n_train_distill=$2
dataset=$3

echo "udf_idx: $udf_idx"
echo "n_train_distill: $n_train_distill"
echo "dataset: $dataset"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/tests/test_distill_model.py \
        --dataset $dataset \
        --udf_idx $udf_idx \
        --labeling_strategy "user" \
        --n_train_distill $n_train_distill