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

### CLEVRER
1. Download the CLEVRER dataset from [here](http://data.csail.mit.edu/clevrer/videos/train/video_train.zip). Place the videos in `data/clevrer/`.
2. Extract the frames from the videos using the following command. This will create a `video_frames` directory in `data/clevrer/`.
```sh
cd data/clevrer
python extract_frames.py
```
3. Prepare the database. Download the annotations from [here](https://drive.google.com/drive/folders/1FBmPlQ1haRCxsmgYMSqaZVPdCp6fcI1m?usp=drive_link) and place them in `duckdb_dir/`.
4. Create relations and load data into the database.
```sh
cd duckdb_dir
python load_clevrer.py
```
5. Extract the features from the frames using the following command. This will create a `features` directory in `data/clevrer/`.
```sh
```

### CityFlow

### Charades

## Example Usage
We provide an example of how to use VOCAL-UDF to process a query with three missing UDFs on the CLEVRER dataset.
1. Generate UDFs
```bash
    python experiments/async_main.py \
        --num_missing_udfs 3 \
        --run_id 0 \
        --query_id 0 \
        --dataset "clevrer" \
        --query_filename "3_new_udfs_labels" \
        --budget 20 \
        --n_selection_samples 500 \
        --num_interpretations 10 \
        --allow_kwargs_in_udf \
        --program_with_pixels \
        --num_parameter_search 5 \
        --num_workers 8 \
        --save_labeled_data \
        --n_train_distill 100 \
        --selection_strategy "both" \
        --llm_method "gpt" \
        --is_async \
        --openai_model_name "gpt-4o"
```

2. Execute query with new UDFs
```bash
    python experiments/run_query_executor.py \
            --num_missing_udfs 3 \
            --run_id 0 \
            --query_id 0 \
            --dataset "clevrer" \
            --query_filename "3_new_udfs_labels" \
            --budget 20 \
            --n_selection_samples 500 \
            --num_interpretations 10 \
            --allow_kwargs_in_udf \
            --program_with_pixels \
            --num_parameter_search 5 \
            --num_workers 8 \
            --n_train_distill 100 \
            --selection_strategy "both" \
            --pred_batch_size 4096 \
            --dali_batch_size 1 \
            --llm_method "gpt"
```

## Reproduce Experiments