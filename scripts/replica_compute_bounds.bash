#!/usr/bin/bash

set -e
# set -x

SCRIPT_DIR=$(cd $(dirname $0) && pwd)

PYTHONPATH=$SCRIPT_DIR/.. python $SCRIPT_DIR/../grad_sdf/dataset/replica_compute_bounds.py \
    --data-path /home/daizhirui/DataArchive/Replica-SDF-aug \
    --max-depth 10
