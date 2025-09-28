#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
#python ${SCRIPT_DIR}/../grad_sdf/dataset/replica_augment_views.py \
#    --original-dir /home/daizhirui/DataArchive/Replica-SDF \
#    --output-dir /home/daizhirui/DataArchive/Replica-SDF-aug \
#    --interval 50

python ${SCRIPT_DIR}/../grad_sdf/dataset/replica_augment_views.py \
    --original-dir /home/daizhirui/DataArchive/Replica-SDF \
    --output-dir /home/daizhirui/DataArchive/Replica-SDF-aug2 \
    --interval 50 \
    --n-rolls-per-insertion 10 \
    --ignore-existing