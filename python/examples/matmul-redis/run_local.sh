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
  --storage_conf=$(pwd)/storage_conf.txt \
  --pythonpath=$(pwd)/src \
  --graph \
  --tracing=$tracing \
  $execFile $@ --check_result --use_storage

  # End the storage standalone backend
  pkill redis
