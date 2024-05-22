#!/bin/bash -l
#SBATCH --job-name=generate
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=23:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

nunsupported_udfs=$1

echo "nunsupported_udfs: $nunsupported_udfs"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/generate_charades_queries.py \
        --n_queries 10 \
        --npred 6 \
        --nobj_pred 2 \
        --nvars 2 \
        --depth 3 \
        --nunsupported_udfs 3