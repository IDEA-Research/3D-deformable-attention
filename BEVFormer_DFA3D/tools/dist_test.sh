#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
# PORT=${PORT:-29503}
function rand(){
        min=20000
        max=$(($60000-$min+1))
        num=$(($RANDOM+1000000000))
        echo $(($num%$max+$min))
}
p=$(rand)
PORT=${PORT:-$p}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
