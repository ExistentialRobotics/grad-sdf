#!/usr/bin/env bash

set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
if [ -z "${APT_MIRROR}"]; then
    docker build --rm -t erl/grad_sdf:24.04 -f ${SCRIPT_DIR}/Dockerfile .
else
    docker build --rm -t erl/grad_sdf:24.04 --build-arg APT_MIRROR=${APT_MIRROR} -f ${SCRIPT_DIR}/Dockerfile .
fi
