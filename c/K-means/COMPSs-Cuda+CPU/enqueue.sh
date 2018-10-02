#!/bin/bash 

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  enqueue_compss \
    --input_profile=/home/bsc19/bsc19430/kmeans-master/COMPSs-Cuda+CPU/prof_N16384_K256 \
    --worker_in_master_cpus=0 \
    --persistent_worker_c=true \
    --lang=c \
    --gpus_per_node=2 \
    --exec_time=5 \
    --num_nodes=2 \
    --cpus_per_node=12 \
    --appdir="${SCRIPT_DIR}" \
    master/kmeans -i "${SCRIPT_DIR}/../generator/N128_K64_d50_0.txt" -n 4 -f 4 -o -l 5

