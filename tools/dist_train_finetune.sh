#!/usr/bin/env bash

CONFIG=$1
# CHECKPOINT=$2
GPUS=$2
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_finetune.py $CONFIG --launcher pytorch ${@:3} --deterministic
