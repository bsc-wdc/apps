#!/bin/bash -e

  # Define script variables
  scriptDir=$(dirname $0)
  execFile=src/neurons.py
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
    --debug \
    $execFile $@


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./run_local.sh tracing num_fragments dataset 
#
# Example:
#       ./run_local.sh false 10 ../data/spikes.dat
#

