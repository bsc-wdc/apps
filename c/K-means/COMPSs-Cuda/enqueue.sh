#!/bin/bash

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  enqueue_compss \
    -d \
    --worker_in_master_cpus=0 \
    --persistent_worker_c=false \
    --lang=c \
    --gpus_per_node=1 \
    --exec_time=5 \
    --num_nodes=2 \
    --cpus_per_node=12 \
    --appdir="${SCRIPT_DIR}" \
    master/kmeans -i "${SCRIPT_DIR}/../generator/N128_K64_d50_0.txt" -n 4 -f 4 -o

