#!/usr/bin/bash

set -e
# set -x

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/../data/Replica_preprocessed}

PYTHONPATH=${SCRIPT_DIR}/.. python "${SCRIPT_DIR}/../grad_sdf/dataset/replica_compute_bounds.py" \
    --data-path "${DATA_DIR}" \
    --max-depth 10
