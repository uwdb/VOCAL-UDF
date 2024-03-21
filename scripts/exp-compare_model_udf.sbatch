#!/bin/bash -l
#SBATCH --job-name=compare_udf
#SBATCH --partition=ckpt
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=4:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

run=$1
udf=$2

echo "run: $run"
echo "udf: $udf"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/compare_code_and_model_udf.py \
        --run_id $run \
        --dataset "clevrer" \
        --udf_class "$udf" \
        --model_based \
        --n_train 100 \
        --save_labeled_data \
        --load_labeled_data