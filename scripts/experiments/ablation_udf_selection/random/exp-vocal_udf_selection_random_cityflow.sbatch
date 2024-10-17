#!/bin/bash -l
#SBATCH --job-name=random_cityflow
#SBATCH --partition=ckpt-all
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

query_id=$1
run=$2
dataset=$3
query_class_name=$4
budget=$5
num_interpretations=$6
num_missing_udfs=$7

echo "query_id: $query_id"
echo "run: $run"
echo "dataset: $dataset"
echo "query_class_name: $query_class_name"
echo "budget: $budget"
echo "num_interpretations: $num_interpretations"
echo "num_missing_udfs: $num_missing_udfs"

bash /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-script.sh \
    python \
        /gscratch/balazinska/enhaoz/VOCAL-UDF/experiments/run_udf_selection_random.py \
        --num_missing_udfs $num_missing_udfs \
        --run_id $run \
        --query_id $query_id \
        --dataset "$dataset" \
        --query_class_name "$query_class_name" \
        --budget $budget \
        --num_interpretations $num_interpretations \
        --allow_kwargs_in_udf \
        --num_parameter_search 5 \
        --generate \
        --cpus 8 \
        --save_labeled_data \
        --n_train_distill 500 \
        --selection_strategy "both" \
        --selection_labels "user" \
        --llm_method "gpt4v" \
        --is_async \
        --openai_model_name "gpt-4o"

# python run_udf_selection_random.py --num_missing_udfs 2 --query_id 0 --run_id 0 --dataset "cityflow" --query_class_name "unavailable_pred=1-unavailable_attr_pred=1-npred=1-nattr_pred=2-nvars=3-depth=3-max_duration=15-min_npos=74-max_npos=737" --budget 50 --num_interpretations 10 --allow_kwargs_in_udf  --num_parameter_search 5  --generate --cpus 8 --save_labeled_data --n_train_distill 500 --selection_strategy "both" --selection_labels "user" --llm_method "gpt4v" --is_async --openai_model_name "gpt-4o"