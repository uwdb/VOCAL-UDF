#!/bin/bash -l
#SBATCH --job-name=exp-llava
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    time python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/tests/test_llava.py \
        --model-path "liuhaotian/llava-v1.5-7b"