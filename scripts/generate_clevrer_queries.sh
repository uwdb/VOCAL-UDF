#!/bin/bash

for nunsupported_udfs in {1..3}; do
    sbatch generate_clevrer_queries.sbatch $nunsupported_udfs
done