#!/usr/bin/env bash

set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
docker build --rm -t erl/grad_sdf:24.04 .
