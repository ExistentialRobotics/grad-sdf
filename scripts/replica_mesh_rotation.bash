#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
python ${SCRIPT_DIR}/../grad_sdf/dataset/replica_obb_rotation.py \
    --dataset-dir /home/daizhirui/DataArchive/Replica-NICE-SLAM \
    --output-dir /home/daizhirui/DataArchive/Replica-SDF
