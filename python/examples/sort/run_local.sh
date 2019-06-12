#!/bin/bash -e

  # Define script variables
  scriptDir=$(dirname $0)
  execFile=src/sort.py
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
#       ./run_local.sh <TRACING> <FILE> <NUM_FRAGMENTS> <NUM_RANGE>
#
# Example:
#       generator/./generate_dataset.sh 102400 200000 $(pwd)/dataset.txt
#       ./run_local.sh false $(pwd)/dataset.txt 5 600
#
