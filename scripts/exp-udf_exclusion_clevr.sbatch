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

method=$1
run=$2
question_id=$3
udf=$4
llm_model=$5

echo "method: $method"
echo "run: $run"
echo "question_id: $question_id"
echo "udf: $udf"
echo "llm_model: $llm_model"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/udf_exclusion_clevr.py \
        --save_output \
        --output_dir "/gscratch/balazinska/enhaoz/VOCAL-UDF/outputs" \
        --run_id $run \
        --question_id $question_id \
        --method "$method" \
        --udf "$udf" \
        --llm_model "$llm_model"