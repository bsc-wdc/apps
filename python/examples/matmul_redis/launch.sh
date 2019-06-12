#!/bin/bash

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/matmul.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/
  # Change this variable if you want to use other storage implementations
  storage_home=${COMPSS_HOME}/Tools/storage/redis

  # Retrieve arguments
  jobDependency=$1
  numNodes=$2
  executionTime=$3
  tasksPerNode=$4
  tracing=$5

  # Leave application args
  shift 5

  # Enqueue the application
  enqueue_compss \
  --job_dependency=$jobDependency \
  --num_nodes=$numNodes \
  --exec_time=$executionTime \
  --max_tasks_per_node=$tasksPerNode \
  --tracing=$tracing \
  --classpath=$appClasspath:${storage_home}/compss-redisPSCO.jar \
  --pythonpath=$appPythonpath \
  --lang=python \
  --storage_home=${storage_home} \
  --storage_props=${storage_home}/scripts/sample_config.cfg \
  $execFile $@

  # Reminder: $@ must contain all the app parameters
  # available app parameters are:
  # -b / --num_blocks Number of blocks (N in NxN)
  # -e / --elems_per_block
  # --check_result Compare distributed product with a sequential one
  # --seed Pseudo-Random seed
  # --use_storage Use storage implementation or not
