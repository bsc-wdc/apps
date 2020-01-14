#!/bin/bash -e

  export COMPSS_PYTHON_VERSION=3-ML
  module load COMPSs/2.6.rc2001
  module load DATACLAY/2.0rc   # remember to add this module load to .bashrc!

  # Retrieve script arguments
  job_dependency=${1:-None}
  num_nodes=${2:-2}
  execution_time=${3:-5}
  tracing=${4:-false}
  exec_file=${5:-$(pwd)/src/matmul.py}

  # Define script variables
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR=${SCRIPT_DIR}/
  APP_CLASSPATH=${SCRIPT_DIR}/src
  APP_PYTHONPATH=${SCRIPT_DIR}/src

  # Define application variables
  graph=$tracing
  log_level="debug"
  qos_flag="--qos=debug"
  workers_flag=""
  constraints=""

  # Create workers sandbox
  # mkdir -p "${WORK_DIR}/COMPSs_Sandbox"
  # --master_working_dir="${WORK_DIR}" \
  # --worker_working_dir="${WORK_DIR}/COMPSs_Sandbox" \

  CPUS_PER_NODE=48
  WORKER_IN_MASTER=24

  shift 5

  # Those are evaluated at submit time, not at start time...
  COMPSS_VERSION=`ml whatis COMPSs 2>&1 >/dev/null | awk '{print $1 ; exit}'`
  DATACLAY_VERSION=`ml whatis DATACLAY 2>&1 >/dev/null | awk '{print $1 ; exit}'`

  # Enqueue job
  enqueue_compss \
    --job_name=matmul_PyCOMPSs_dataClay \
    --job_dependency="${job_dependency}" \
    --exec_time="${execution_time}" \
    --num_nodes="${num_nodes}" \
    \
    --cpus_per_node="${CPUS_PER_NODE}" \
    --worker_in_master_cpus="${WORKER_IN_MASTER}" \
    \
    "${workers_flag}" \
    \
    --worker_working_dir=scratch \
    \
    --constraints=${constraints} \
    --tracing="${tracing}" \
    --graph="${graph}" \
    --summary \
    --log_level="${log_level}" \
    "${qos_flag}" \
    \
    --classpath=$DATACLAY_JAR:$DATACLAY_DEPENDENCY_LIBS/*:${APP_CLASSPATH}:${CLASSPATH} \
    --pythonpath=${APP_PYTHONPATH}:${PYTHONPATH} \
    --storage_props=$(pwd)/dataClay_confs/storage_props.cfg \
    --storage_home=$COMPSS_STORAGE_HOME \
    --prolog=$(pwd)/dataClay_confs/register.sh \
    \
    --lang=python \
    \
    "$exec_file" $@

# Enqueue tests example:
# ./launch_with_dataClay.sh None 2 5 false $(pwd)/src/matmul.py -b 4 -e 4 --check_result

# OUTPUTS:
# - compss-XX.out : Job output file
# - compss-XX.err : Job error file
# - ~/.COMPSs/JOB_ID/ : COMPSs files
