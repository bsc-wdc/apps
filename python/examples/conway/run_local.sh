#!/bin/bash -e

  # Define script variables
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  LOCAL_PYTHONPATH=${SCRIPT_DIR}/src/
  EXEC_FILE=${SCRIPT_DIR}/src/conway.py
  
  # Retrieve arguments
  tracing=${1:-false}
  graph=${2:-false}
  log_level=${3:-debug}

  # Ensure output directory exists
  mkdir -p "${OUTPUT_DIR}"

  # Run Job
  runcompss \
    --tracing="${tracing}" \
    --graph="${graph}" \
    --summary \
    --log_level="${log_level}" \
    \
    --pythonpath="${LOCAL_PYTHONPATH}" \
    --lang=python \
    \
    "${EXEC_FILE}" 32 32 40 16 9


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./run_local.sh <TRACING> <GRAPH> <LOG_LEVEL>
#
# Example:
#       ./run_local.sh false false debug
#
