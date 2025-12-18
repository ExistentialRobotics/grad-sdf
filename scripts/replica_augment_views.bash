#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/../data/Replica_preprocessed}
INTERVAL=${INTERVAL:-50}
N_ROLLS_PER_INSERTION=${N_ROLLS_PER_INSERTION:-15}
MAX_ROLL_OF_INSERTION=${MAX_ROLL_OF_INSERTION:-3.1415926}

python ${SCRIPT_DIR}/../grad_sdf/dataset/replica_augment_views.py \
    --original-dir "${DATA_DIR}" \
    --output-dir "${DATA_DIR}" \
    --interval ${INTERVAL} \
    --n-rolls-per-insertion ${N_ROLLS_PER_INSERTION} \
    --max-roll-of-insertion ${MAX_ROLL_OF_INSERTION}
