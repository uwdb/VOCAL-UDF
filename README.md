# VOCAL-UDF

A prototype implementation of VOCAL-UDF, which is a self-enhancing video data management system that empowers users to flexibly issue and answer compositional queries, even when the modules necessary to answer those queries are unavailable.

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

## Prepare Data

TODO

## Run experiment
```bash
cd scripts
```
1. Generate UDFs
```bash
./exp-vocal_udf_main_{dataset}.sh
```

2. Execute query with new UDFs
```bash
./exp-vocal_udf_query_execution_{dataset}.sh
```
