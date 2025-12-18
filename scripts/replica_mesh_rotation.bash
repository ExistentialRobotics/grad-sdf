#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/../data/Replica}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/../data/Replica_preprocessed}

PYTHON_PATH=${SCRIPT_DIR}/.. python ${SCRIPT_DIR}/../grad_sdf/dataset/replica_obb_rotation.py \
    --dataset-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR}
