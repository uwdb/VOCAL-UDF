#!/bin/bash -l
#SBATCH --job-name=exp-visprog_clevr
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=4:00:00
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
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/visprog_clevr.py \
        --use_precomputed \
        --save_output \
        --output_dir "/gscratch/balazinska/enhaoz/VOCAL-UDF/outputs/clevr" \
        --run_id $run \
        --question_id $question_id \
        --task_name "$task_name" \