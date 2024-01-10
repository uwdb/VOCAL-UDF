#!/bin/bash

# Example usage:
# ./singularity-script.sh <target-script>.sh <target-script args>
# The target script will be executed inside of the container.

# module load apptainer
# /gscratch/balazinska/enhaoz/apptainer/launch-container-ro.sh /gscratch/balazinska/enhaoz/sbatch/apptainer-env-setup.sh "${@}"
apptainer exec --nv --bind /scr,/gscratch --overlay /gscratch/balazinska/enhaoz/apptainer/conda-overlay.img:ro /gscratch/balazinska/enhaoz/apptainer/hyak-container.sif /gscratch/balazinska/enhaoz/VOCAL-UDF/scripts/apptainer-env-setup.sh "${@}"
# apptainer exec --nv --bind /scr,/gscratch /gscratch/balazinska/enhaoz/apptainer/tools.sif /gscratch/balazinska/enhaoz/sbatch/apptainer-env-setup.sh "${@}"
