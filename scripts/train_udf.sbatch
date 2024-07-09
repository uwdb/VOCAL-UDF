#!/bin/bash -l
#SBATCH --job-name=train_udf
#SBATCH --partition=gpu-a40
#SBATCH --account=krishna
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --time=1:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"
dataset=$1
udf_name=$2

echo "dataset: $dataset"
echo "udf_name: $udf_name"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/prediction/train_udf.py \
        --dataset $dataset \
        --udf_name $udf_name \
        --max_positive_samples 5000 \
        --balanced