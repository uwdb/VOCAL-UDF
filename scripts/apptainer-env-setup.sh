#!/bin/bash
# We can't install it to the site-packages directory because we don't have write access after the container is built.
# Work around by installing to a local directory that is on the python path.

# source /opt/conda/etc/profile.d/conda.sh
# conda activate env
. ~/.bashrc
conda activate /opt/miniconda/env

if [[ $# -eq 0 ]]; then
    /bin/bash
else
    "${@}"
fi