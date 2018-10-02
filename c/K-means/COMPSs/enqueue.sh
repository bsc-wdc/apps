#!/bin/bash

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  enqueue_compss \
    -d \
    --persistent_worker_c=true \
    --lang=c \
    --exec_time=10 \
    --num_nodes=2 \
    --cpus_per_node=12 \
    --appdir="${SCRIPT_DIR}" \
    master/kmeans -i "${SCRIPT_DIR}/../generator/N128_K64_d50_0.txt" -n 4 -f 4 -o -l 10

