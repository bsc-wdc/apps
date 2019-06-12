#!/bin/bash -e

  # Define script variables
  scriptDir=$(dirname $0)
  execFile=${scriptDir}/src/wordcount.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  tracing=$1

  # Leave application args on $@
  shift 1

  # Enqueue the application
  runcompss \
    --tracing=$tracing \
    --graph=true \
    --classpath=$appClasspath \
    --pythonpath=$appPythonpath \
    --lang=python \
    $execFile $@


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./run_local.sh <TRACING> <DATASET_PATH> <MULTIPLE_FILES>
#
# Example:
#       # Create a dataset before running it
#       ./run_local.sh /path/to/dataset/ True
#
