#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/max.py
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

  # Enqueue the application
  enqueue_compss \
    --job_dependency=$jobDependency \
    --num_nodes=$numNodes \
    --exec_time=$executionTime \
    --tasks_per_node=$tasksPerNode \
    --master_working_dir=. \
    --worker_working_dir=scratch \
    --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
    --tracing=$tracing \
    --classpath=$appClasspath \
    --pythonpath=$appPythonpath \
    --lang=python \
    $execFile $@


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./launch jobDependency numNodes executionTime tasksPerNode tracing numP sim numFrag 
#
# Example:
#       ./launch None 2 5 16 false 16000 3 16
#
