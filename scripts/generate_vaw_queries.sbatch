#!/bin/bash -l
#SBATCH --job-name=generate
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=5:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

nunsupported_udfs=$1

echo "nunsupported_udfs: $nunsupported_udfs"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/generate_vaw_queries.py \
        --n_queries 20 \
        --npred 2 \
        --nattr_pred 1 \
        --nvars 3 \
        --nunsupported_pred 1 \
        --nunsupported_attr_pred 1 \
        --dataset vaw \
        --min_npos 3000 \
        --max_npos 20000

# python generate_vaw_queries.py --n_queries 20 --npred 2 --nattr_pred 1 --nvars 3 --nunsupported_pred 1 --nunsupported_attr_pred 1 --dataset vaw --min_npos 3000 --max_npos 20000