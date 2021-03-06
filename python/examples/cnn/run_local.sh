#!/bin/bash -e

  # Define script variables
  scriptDir=$(dirname $0)
  execFile=src/compss_mnist.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  tracing=$1

  # Leave application args on $@
  shift 1

  export LC_ALL="en_US.UTF-8"

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
#       ./run_local.sh <TRACING> <BASE_PATH> <NUM_MODELS>
#
# Example:
#       ./run_local.sh false . 2
#
