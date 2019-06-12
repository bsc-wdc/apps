#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=src/pycompss_modelfactors.py
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
#       ./run_local.sh <TRACING> <TRACES_PATH> --cfgs=<CFGS_PATH> --out=<OUTPUT_PATH> -d
#
# Example:
#       ./run_local.sh false /path/to/traces/* --cfgs=/path/to/modelfactors/cfgs/ --out=/path/to/output -d
#
