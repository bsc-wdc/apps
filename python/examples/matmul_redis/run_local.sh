#!/bin/bash -e

  # Define script variables
  scriptDir=$(dirname $0)
  execFile=src/matmul.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  tracing=$1

  # Leave application args $@
  shift 1

  # Set a standalone Redis backend
  redis-server --daemonize yes

  # Launch runcompss with all the appropriate arguments
  runcompss --lang=python \
  --storage_impl=redis \
  --storage_conf=${scriptDir}/storage_conf.txt \
  --classpath=${appClasspath} \
  --pythonpath=${appPythonpath} \
  --graph \
  --tracing=$tracing \
  $execFile $@ --check_result --use_storage

  # End the storage standalone backend
  pkill redis

  ######################################################
  # APPLICATION EXECUTION EXAMPLE
  # Call:
  #       ./run_local.sh <TRACING> -b <NUM_BLOCKS> -e <NUM_ELEMENTS>
  #
  # Example:
  #       ./run_local.sh false -b 4 -e 4
  #
