#!/bin/bash -l
#SBATCH --job-name=exp-vocal_udf_main
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

run=$1
query_id=$2
dataset=$3
budget=$4
num_interpretations=$5

echo "run: $run"
echo "query_id: $query_id"
echo "dataset: $dataset"
echo "budget: $budget"
echo "num_interpretations: $num_interpretations"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/vocaludf/main.py \
        --run_id $run \
        --query_id $query_id \
        --dataset "$dataset" \
        --budget $budget \
        --num_interpretations $num_interpretations