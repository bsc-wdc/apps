#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/cholesky.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  jobDependency=$1
  numNodes=$2
  executionTime=$3
  tasksPerNode=$4
  tracing=$5

  # Leave application args on $@
  shift 5

  export ComputingUnits="1"

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
#       ./launch.sh jobDependency numNodes executionTime tasksPerNode tracing MSIZE BSIZE MKL_NUM_THREADS
#
# Example:
#       ./launch.sh None 2 10 48 true 4 512 8
#