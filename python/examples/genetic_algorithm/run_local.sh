#!/bin/bash -e

  # Define script variables
  scriptDir=$(dirname $0)
  execFile=src/gen.py
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
#       ./run_local.sh <TRACING> <NUM_IND> <SIZE_IND> <TARGET> <CYCLES>
#
# Example:
#       ./run_local.sh false 10 10 20 10

#
