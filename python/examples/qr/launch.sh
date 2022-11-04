#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/qr.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  jobDependency=$1
  numNodes=$2
  executionTime=$3
  tracing=$4
  computingUnits=$5

  # Leave application args on $@
  shift 5

  export ComputingUnits="${computingUnits}"

  # Enqueue the application
  enqueue_compss \
    --job_dependency=$jobDependency \
    --num_nodes=$numNodes \
    --exec_time=$executionTime \
    --job_execution_dir=. \
    --worker_working_dir=gpfs \
    --tracing=$tracing \
    --pythonpath=$appPythonpath \
    --lang=python \
    $execFile $@


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./launch.sh <JOB_DEPENDENCY> <NUM_NODES> <EXECUTION_TIME> <TRACING> <COMPUTING_UNITS> <MSIZE> <BSIZE> <MKL_NUM_THREADS> <VERIFY_RESULT>
#
# Example:
#       ./launch.sh None 2 10 true 1 4 512 1 False
#
