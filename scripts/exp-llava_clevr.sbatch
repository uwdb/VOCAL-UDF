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

task_name=$1
run=$2
question_id=$3

echo "task_name: $task_name"
echo "run: $run"
echo "question_id: $question_id"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    time python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/llava_clevr.py \
        --save_output \
        --output_dir "/gscratch/balazinska/enhaoz/VOCAL-UDF/outputs" \
        --run_id $run \
        --question_id $question_id \
        --task_name "$task_name" \
        --batch_size 32

# bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
#     time python \
#         /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/llava_clevr.py \
#         --save_output \
#         --output_dir "/gscratch/balazinska/enhaoz/VOCAL-UDF/outputs" \
#         --run_id $run \
#         --question_id $question_id \
#         --task_name "$task_name" \
#         --use_ccot \
#         --batch_size 16