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
  --classpath=$appClasspath \
  --pythonpath=$appPythonpath \
  --tracing=$tracing \
  -t \
  -g \
  $execFile $@

  # Kill the storage backend
  pkill redis

#####################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./run_local_less.sh <TRACING> -n <NUM_POINTS> -d <DIMENSIONS> -c <CENTRES> -f <FRAGMENTS>
#
# Example:
#       ./run_local_less.sh false -n 160 -d 3 -c 4 -f 4
#
