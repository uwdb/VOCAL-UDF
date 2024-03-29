#!/bin/bash -l
#SBATCH --job-name=llm_labels
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

run=$1
udf=$2
method=$3

echo "run: $run"
echo "udf: $udf"
echo "method: $method"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/llm_labels_relationship.py \
        --run_id $run \
        --dataset "gqa" \
        --relationship "$udf" \
        --n_train 100 \
        --method "$method" \
        --save_labeled_data

# --load_labeled_data \