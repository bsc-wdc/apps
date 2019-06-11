#!/usr/bin/env bash

# This script runs the Redis application with the storage_init/stop scripts
# These scripts build and shut down a Redis cluster in a totally automated way

# Storage classpath plus storage dependencies
storage_home=$(pwd)/COMPSs-Redis-bundle
storage_classpath=${storage_home}/compss-redisPSCO.jar
WORKER_WORKING_DIR=/mnt/lustre/Computational/COMPSs/srodrig1/

small_params="50 5 5 1 --seed 1 --check_result"
big_params="1000000 100 50 1000 --seed 1 --check_result"

time \
enqueue_compss \
  --job_dependency=None \
  --num_nodes=3 \
  --cpus_per_node=16 \
  --exec_time=20 \
  --master_working_dir=. \
  --worker_working_dir=${WORKER_WORKING_DIR} \
  --pythonpath=$storage_home/python:$(pwd) \
  --classpath=$exec_classpath:$storage_classpath \
  --storage_props=$storage_home/scripts/sample_props.cfg \
  --storage_home=$storage_home/ \
  $(pwd)/main.py $small_params

