#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/kmeans.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/
  storage_home=$(pwd)/COMPSs-Redis-bundle
  storage_classpath=${storage_home}/compss-redisPSCO.jar
  WORKER_WORKING_DIR=scratch
  WORK_DIR=${scriptDir}/results/


  # Retrieve arguments
  jobDependency=$1
  numNodes=$2
  executionTime=$3
  tasksPerNode=$4
  tracing=$5

  # Leave application args on $@
  shift 5

  export ComputingUnits=12

  # Enqueue the application
  time \
  enqueue_compss \
    --exec_time=120 \
    --job_dependency=$jobDependency \
    --num_nodes=$numNodes \
    --cpus_per_node=$tasksPerNode \
    --exec_time=$executionTime \
    --master_working_dir=$WORK_DIR \
    --worker_working_dir=$WORKER_WORKING_DIR \
    --classpath=$appClasspath:$storage_classpath \
    --pythonpath=$appPythonpath:$storage_home/python \
    --storage_props=$storage_home/scripts/sample_props.cfg \
    --storage_home=$storage_home/ \
    --qos=debug \
    --lang=python \
    --tracing \
    $execFile $@


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./launch.sh jobDependency numNodes executionTime tasksPerNode tracing numV sim k numFrag
#
# Example:
#       ./launch.sh None 2 5 16 false 16000 3 4 16
#
#  numV = numero de punts
#  dim = dimensions del punt
#  k = numero de centres
#  numFrag = numero de fragments 
