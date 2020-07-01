#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/bioinf.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  numNodes=$1
  executionTime=$2
  tracing=$3

  # Leave application args on $@
  shift 3

  # Enqueue the application
  enqueue_compss \
     --num_nodes=$numNodes \
     --exec_time=$executionTime \
     --master_working_dir=. \
     --worker_working_dir=gpfs \
     --classpath=$appClasspath \
     --pythonpath=$appPythonpath \
     --tracing=$tracing \
     --lang=python \
     $execFile $@


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./launch.sh <NUM_NODES> <EXECUTION_TIME> <TRACING>
#
# Example:
#       ./launch.sh 2 10 false
