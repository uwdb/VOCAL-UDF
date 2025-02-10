#!/bin/bash -l
#SBATCH --job-name=gpt_clevrer
#SBATCH --partition=gpu-l40s
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

query_filename=$1
run=$2
query_id=$3

echo "query_filename: $query_filename"
echo "run: $run"
echo "query_id: $query_id"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/gpt_clevrer_simplified.py \
        --run_id $run \
        --query_id $query_id \
        --query_filename "$query_filename" \
        --openai_model_name "gpt-4o" \
        --stage "retrieve"


# --stage "submit"
# --stage "retrieve"