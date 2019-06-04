#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=src/terasort.py
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
#       ./run.sh tracing numFragments numEntries
#
# Example:
#       ./launch.sh false 10 100
#