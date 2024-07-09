# VOCAL-UDF

## Virtual environment

```bash
conda activate equi-vocal
cd EQUI-VOCAL/
python -m pip install -e .
```

## Run experiment

# VOCAL-UDF
1. Generate UDFs
```bash
./exp-vocal_udf_main_{dataset}.sh
```

2. Execute query with new UDFs
```bash
./exp-vocal_udf_query_execution_{dataset}.sh
```

# EQUI-VOCAL
```bash
./exp-evaluate_equivocal.sh
```

# VisProg
```bash
./exp-visprog.sh
```