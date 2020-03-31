#!/bin/bash -e

  export COMPSS_PYTHON_VERSION=3-ML
  module load COMPSs/2.6.3
  module load ruby

  export PATH=$(pwd)/../redis/:$PATH

  # Storage-related paths
  # Change these paths if you want to use other storage implementations
  STORAGE_HOME=${COMPSS_HOME}/Tools/storage/redis
  STORAGE_CLASSPATH=${STORAGE_HOME}/compss-redisPSCO.jar

  # Retrieve script arguments
  job_dependency=${1:-None}
  num_nodes=${2:-2}
  execution_time=${3:-5}
  tracing=${4:-false}
  exec_file=${5:-$(pwd)/src/wordcount_storage.py}

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

  # Enqueue job
  enqueue_compss \
    --job_name=matmul_PyCOMPSs_redis \
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
    --classpath=${APP_CLASSPATH}:${STORAGE_CLASSPATH}:${CLASSPATH} \
    --pythonpath=${APP_PYTHONPATH}:${STORAGE_HOME}/python:${PYTHONPATH} \
    --storage_props=$(pwd)/redis_confs/storage_props.cfg \
    --storage_home=${STORAGE_HOME}/ \
    \
    --lang=python \
    \
    "$exec_file" $@

# Enqueue tests example:
# ./launch_with_redis.sh None 2 5 false $(pwd)/src/wordcount_storage.py -d $(pwd)/dataset

# OUTPUTS:
# - compss-XX.out : Job output file
# - compss-XX.err : Job error file
# - ~/.COMPSs/JOB_ID/ : COMPSs files
