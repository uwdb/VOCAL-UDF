# VOCAL-UDF

A prototype implementation of VOCAL-UDF, which is a self-enhancing video data management system that empowers users to flexibly issue and answer compositional queries, even when the modules necessary to answer those queries are unavailable. See the [technical report](https://arxiv.org/pdf/2408.02243) for more details.

## Setup Instructions

The project uses `conda` to manage dependencies. To install conda, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

```sh
# Clone the repository
git clone https://github.com/uwdb/VOCAL-UDF.git
cd VOCAL-UDF

# Create a conda environment (called vocal-udf) and install dependencies
conda env create -f environment.yml --name vocal-udf
conda activate vocal-udf
python -m pip install -e .
```

To use OpenAI models, follow the instructions [here](https://platform.openai.com/docs/quickstart#create-and-export-an-api-key) to create and export an API key.
```sh
# Export the API key as an environment variable
export OPENAI_API_KEY="your_api_key_here"
```

## Prepare Data

TODO

## Example Usage
```bash
cd scripts
```
1. Generate UDFs
```bash
    python experiments/run_query_executor.py \
            --num_missing_udfs $num_missing_udfs \
            --run_id $run \
            --query_id $query_id \
            --dataset "$dataset" \
            --query_class_name "$query_class_name" \
            --budget $budget \
            --n_selection_samples 500 \
            --num_interpretations $num_interpretations \
            --allow_kwargs_in_udf \
            --program_with_pixels \
            --num_parameter_search 5 \
            --cpus 8 \
            --n_train_distill 100 \
            --selection_strategy "both" \
            --selection_labels "user" \
            --pred_batch_size 4096 \
            --dali_batch_size 1 \
            --llm_method "gpt4v"
```

2. Execute query with new UDFs
```bash
./exp-vocal_udf_query_execution_{dataset}.sh
```

## Reproduce Experiments