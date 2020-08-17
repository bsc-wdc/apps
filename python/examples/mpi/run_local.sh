#!/bin/bash -e

  # Define script variables
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  LOCAL_PYTHONPATH=${SCRIPT_DIR}/src/
  EXEC_FILE=${SCRIPT_DIR}/src/mpi_manager.py
  OUTPUT_DIR=${SCRIPT_DIR}/output/
  BIN_DIR=${SCRIPT_DIR}/bin/
  
  # Retrieve arguments
  cus=${1:-4}
  tracing=${2:-false}
  graph=${3:-false}
  log_level=${4:-debug}

  # Compile MPI
  (
    cd "${BIN_DIR}"
    mpicc -o hello_world.x hello_world.c
    mpicc -o parameters.x parameters.c
    mpicc -o complex.x complex.c 
  )

  # Ensure output directory exists
  mkdir -p "${OUTPUT_DIR}"

  # Export variables for workflow
  export MPI_TASK_NUM_NODES=2
  export MPI_TASK_NUM_CUS=${cus}
  export BIN_DIR=${BIN_DIR}
  export OUTPUT_DIR=${OUTPUT_DIR}

  # Run Job
  runcompss \
    --tracing="${tracing}" \
    --graph="${graph}" \
    --summary \
    --log_level="${log_level}" \
    \
    --project="${SCRIPT_DIR}/xmls/project.xml" \
    --resources="${SCRIPT_DIR}/xmls/resources.xml" \
    \
    --base_log_dir="${OUTPUT_DIR}" \
    --pythonpath="${LOCAL_PYTHONPATH}" \
    --lang=python \
    \
    "${EXEC_FILE}"


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./run_local.sh <COMPUTING_UNITS> <TRACING> <GRAPH> <LOG_LEVEL>
#
# Example:
#       ./run_local.sh 4 false false debug
#
