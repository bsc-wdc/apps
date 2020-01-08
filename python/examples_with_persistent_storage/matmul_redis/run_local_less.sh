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
  --pythonpath=${scriptDir}/src \
  --graph \
  --tracing=$tracing \
  $execFile $@ --check_result

  ######################################################
  # APPLICATION EXECUTION EXAMPLE
  # Call:
  #       ./run_local_less.sh <TRACING> -b <NUM_BLOCKS> -e <NUM_ELEMENTS>
  #
  # Example:
  #       ./run_local_less.sh false -b 4 -e 4
  #
