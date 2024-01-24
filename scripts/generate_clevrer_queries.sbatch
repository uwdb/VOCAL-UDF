#!/bin/bash -l
#SBATCH --job-name=generate_clevrer_queries
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=23:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

nunsupported_udfs=$1

echo "nunsupported_udfs: $nunsupported_udfs"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/generate_clevrer_queries.py \
        --nunsupported_udfs $nunsupported_udfs \
        --max_workers 1