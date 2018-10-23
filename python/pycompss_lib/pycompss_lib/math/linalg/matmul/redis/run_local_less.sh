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

  # Launch runcompss with all the appropriate arguments
  runcompss --lang=python \
  --pythonpath=$(pwd)/src \
  --graph \
  --tracing=$tracing \
  $execFile $@ --check_result

