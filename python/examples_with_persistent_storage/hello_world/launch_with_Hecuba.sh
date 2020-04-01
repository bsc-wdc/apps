#!/bin/bash -e

  export COMPSS_PYTHON_VERSION=3
  module load COMPSs/2.6
  module load hecuba/0.1.3

  # Retrieve script arguments
  job_dependency=${1:-None}
  num_nodes=${2:-2}
  execution_time=${3:-5}
  tracing=${4:-false}
  exec_file=${5:-$(pwd)/src/hello_world.py}

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
  HECUBA_VERSION=`ml whatis HECUBA 2>&1 >/dev/null | awk '{print $1 ; exit}'`

  # Enqueue job
  enqueue_compss \
    --job_name=helloworld_PyCOMPSs_Hecuba \
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
    --classpath=$HECUBA_ROOT/storage_home/StorageItf-1.0-jar-with-dependencies.jar:${APP_CLASSPATH}:${CLASSPATH} \
    --pythonpath=${APP_PYTHONPATH}:${PYTHONPATH} \
    --storage_props=$(pwd)/hecuba_confs/storage_props.cfg \
    --storage_home=$HECUBA_ROOT/ \
    \
    --lang=python \
    \
    "$exec_file" $@

# --prolog=$(pwd)/hecuba_confs/register.sh \
# --classpath=/home/bsc31/bsc31906/hecuba-src/storageLtf/StorageItf-1.0-jar-with-dependencies.jar:${APP_CLASSPATH}:${CLASSPATH} \


# Enqueue tests example:
# ./launch_with_Hecuba.sh None 2 5 false $(pwd)/src/hello_world.py

# OUTPUTS:
# - compss-XX.out : Job output file
# - compss-XX.err : Job error file
# - ~/.COMPSs/JOB_ID/ : COMPSs files
