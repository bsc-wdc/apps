#!/bin/bash -e

  # Define script variables
  scriptDir=$(dirname $0)
  execFile=src/max_norm.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  tracing=$1

  # Leave application args on $@
  shift 1

  # Enqueue the application
  runcompss \
    --tracing=$tracing \
    --classpath=$appClasspath \
    --pythonpath=$appPythonpath \
    --lang=python \
    $execFile $@

######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./run_local.sh <TRACING> <POINTS> <DIM> <FRAGMENTS>
#
# Example:
#       ./run_local.sh false 16000 3 16
#
