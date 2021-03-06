#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=src/cholesky.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  tracing=$1
  ComputingUnits=$2

  # Leave application args on $@
  shift 2

  export ComputingUnits=${ComputingUnits}

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
#       ./run_local.sh <TRACING> <COMPUTING_UNITS> <MSIZE> <BSIZE> <MKL_NUM_THREADS>
#
# Example:
#       ./run_local.sh false 1 4 8 4
#
