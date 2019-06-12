#!/bin/bash -e

  # Define script variables
  scriptDir=$(dirname $0)
  execFile=src/kmeans.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/


  # Retrieve arguments
  tracing=$1

  # Leave the application args on $0
  shift 1

  # Init the storage backend
  redis-server --daemonize yes

  # Launch the application
  runcompss \
  --storage_impl=redis \
  --classpath=$appClasspath \
  --pythonpath=$appPythonpath \
  --tracing=$tracing \
  --storage_conf=${scriptDir}/storage_conf.txt \
  -t \
  -g \
  $execFile $@ --use_storage

  # Kill the storage backend
  pkill redis

#####################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./run_local.sh <TRACING> -n <NUM_POINTS> -d <DIMENSIONS> -c <CENTRES> -f <FRAGMENTS>
#
# Example:
#       ./run_local.sh false -n 160 -d 3 -c 4 -f 4
#
