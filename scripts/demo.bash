#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
SRC_DIR=$(cd ${SCRIPT_DIR}/..; pwd)
CONFIG_DIR=${SRC_DIR}/../configs
REPLICA_DATA_DIR=${HOME}/DataArchive/Replica-SDF-aug

PYTHONPATH=${SCRIPT_DIR}/.. python3 ${SRC_DIR}/grad_sdf/gui_trainer.py \
    --gui-config ${CONFIG_DIR}/gui.yaml \
    --trainer-config ${CONFIG_DIR}/replica_room0.yaml \
    --gt-mesh-path ${REPLICA_DATA_DIR}/room0_mesh.ply \
    --apply-offset-to-gt-mesh \
    --copy-scene-bound-to-gui
