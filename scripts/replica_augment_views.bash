#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/../data/Replica_preprocessed}

python ${SCRIPT_DIR}/../grad_sdf/dataset/replica_augment_views.py \
    --original-dir "${DATA_DIR}" \
    --output-dir "${DATA_DIR}" \
    --interval 50 \
    --n-rolls-per-insertion 10 \
    --max-roll-of-insertion 1.5707963268
