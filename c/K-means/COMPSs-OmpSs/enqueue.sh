#!/bin/bash

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  enqueue_compss \
    -d \
    -m \
    --lang=c \
    --persistent_worker_c=true \
    --exec_time=20 \
    --num_nodes=2 \
    --worker_in_master_cpus=0 \
    --cpus_per_node=12 \
    --appdir="${SCRIPT_DIR}" \
    master/kmeans -i "${SCRIPT_DIR}/../generator/N128_K64_d50_0.txt" -n 4 -f 4 -o -l 25

