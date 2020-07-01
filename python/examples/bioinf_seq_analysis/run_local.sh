#!/bin/bash -e

  # Define script variables
  scriptDir=$(dirname $0)
  execFile=src/bioinf.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  tracing=$1

  echo $appClasspath
  echo $appPythonpath

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
#       ./run_local.sh <TRACING>
#
# Example:
#       ./run_local.sh false
#
