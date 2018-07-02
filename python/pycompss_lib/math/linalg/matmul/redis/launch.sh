#!/bin/bash

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/matmul.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

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
  --tasks_per_node=$tasksPerNode \
  --tracing=$tracing \
  --classpath=$appClasspath \
  --pythonpath=$appPythonpath \
  --lang=python \
  --storage_home=$(pwd)/Redis \
  --storage_home=$(pwd)/sample_props.cfg \
  $execFile $@
