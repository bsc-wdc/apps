#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/neurons.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  jobDependency=$1
  numNodes=$2
  executionTime=$3
  tracing=$4

  # Leave application args on $@
  shift 4

  # Enqueue the application
  enqueue_compss \
    --job_dependency=$jobDependency \
    --num_nodes=$numNodes \
    --exec_time=$executionTime \
    --job_execution_dir=. \
    --worker_working_dir=gpfs \
    --tracing=$tracing \
    --lang=python \
    $execFile $@


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./launch.sh <JOB_DEPENDENCY> <NUM_NODES> <EXECUTION_TIME> <TRACING> <NUM_FRAGMENTS> <DATASET_PATH>
#
# Example:
#       ./launch.sh None 2 10 false 1024 $(pwd)/data/spikes.dat
#
