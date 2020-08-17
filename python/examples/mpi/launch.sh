#!/bin/bash -e

  # Define script variables
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  LOCAL_PYTHONPATH=${SCRIPT_DIR}/src/
  EXEC_FILE=${SCRIPT_DIR}/src/mpi_manager.py
  OUTPUT_DIR=${SCRIPT_DIR}/output/
  BIN_DIR=${SCRIPT_DIR}/bin/

  # Retrieve arguments
  job_dependency=${1:-None}
  num_nodes=${2:-3}
  execution_time=${3:-15}
  cpus_per_node=${4:-48}
  tracing=${5:-false}
  graph=${6:-false}
  log_level=${7:-debug}

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
  export MPI_TASK_NUM_CUS=48
  export BIN_DIR=${BIN_DIR}
  export OUTPUT_DIR=${OUTPUT_DIR}

  # Enqueue job
  enqueue_compss \
    --job_dependency="${job_dependency}" \
    --exec_time="${execution_time}" \
    --num_nodes="${num_nodes}" \
    \
    --cpus_per_node="${cpus_per_node}" \
    --worker_in_master_cpus=0 \
    \
    --tracing="${tracing}" \
    --graph="${graph}" \
    --summary \
    --log_level="${log_level}" \
    \
    --master_working_dir="${OUTPUT_DIR}" \
    --worker_working_dir="${OUTPUT_DIR}" \
    --base_log_dir="${OUTPUT_DIR}" \
    --pythonpath="${LOCAL_PYTHONPATH}" \
    --lang=python \
    \
    "${EXEC_FILE}"


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./launch.sh <JOB_DEPENDENCY> <NUM_NODES> <EXECUTION_TIME> <CPUS_PER_NODE> <TRACING> <GRAPH> <LOG_LEVEL>
#
# Example:
#       ./launch.sh None 3 15 48 false false debug
#
