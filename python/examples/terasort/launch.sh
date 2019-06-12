#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/terasort.py
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
    --master_working_dir=. \
    --worker_working_dir=gpfs \
    --tracing=$tracing \
    --classpath=$appClasspath \
    --pythonpath=$appPythonpath \
    --lang=python \
    $execFile $@


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./launch.sh <JOB_DEPENDENCY> <NUM_NODES> <EXECUTION_TIME> <TRACING> <NUM_FRAGMENTS> <NUM_ENTRIES>
#
# Example:
#       ./launch.sh None 2 5 false 10 100
#
