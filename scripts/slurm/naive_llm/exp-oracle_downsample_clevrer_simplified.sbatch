#!/bin/bash -l
#SBATCH --job-name=oracle_downsample
#SBATCH --partition=ckpt
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"


query_id=$1
sampling_rate=$2

echo "query_id: $query_id"
echo "sampling_rate: $sampling_rate"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/oracle_downsample_clevrer_simplified.py \
        --query_id $query_id \
        --query_filename "simplified_3_new_udfs_labels" \
        --sampling_rate $sampling_rate