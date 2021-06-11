#!/bin/bash -e

  export COMPSS_PYTHON_VERSION=3
  module load 2.9.pr

  # Retrieve script arguments
  job_dependency=${1:-None}
  num_nodes=${2:-2}
  execution_time=${3:-5}
  tracing=${4:-false}
  exec_file=${5:-$(pwd)/src/kmeans.py}

  # Define script variables
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR=${SCRIPT_DIR}/
  APP_CLASSPATH=${SCRIPT_DIR}/src
  APP_PYTHONPATH=${SCRIPT_DIR}/src

  # Define application variables
  graph=$tracing
  log_level="off"
  qos_flag="--qos=debug"
  workers_flag=""
  constraints="highmem"

  # Create workers sandbox
  # mkdir -p "${WORK_DIR}/COMPSs_Sandbox"
  # --master_working_dir="${WORK_DIR}" \
  # --worker_working_dir="${WORK_DIR}/COMPSs_Sandbox" \

  CPUS_PER_NODE=48
  WORKER_IN_MASTER=0

  shift 5

  # Those are evaluated at submit time, not at start time...
  COMPSS_VERSION=`ml whatis COMPSs 2>&1 >/dev/null | awk '{print $1 ; exit}'`

  # Enqueue job
  enqueue_compss \
    --job_name=kmeans_PyCOMPSs \
    --job_dependency="${job_dependency}" \
    --exec_time="${execution_time}" \
    --num_nodes="${num_nodes}" \
    \
    --cpus_per_node="${CPUS_PER_NODE}" \
    --worker_in_master_cpus="${WORKER_IN_MASTER}" \
    --scheduler=es.bsc.compss.scheduler.fifodatalocation.FIFODataLocationScheduler \
    \
    "${workers_flag}" \
    \
    --worker_working_dir=local_disk \
    \
    --constraints=${constraints} \
    --tracing="${tracing}" \
    --graph="${graph}" \
    --summary \
    --log_level="${log_level}" \
    "${qos_flag}" \
    \
    --classpath=${APP_CLASSPATH}:${CLASSPATH} \
    --pythonpath=${APP_PYTHONPATH}:${PYTHONPATH} \
    \
    --lang=python \
    \
    "$exec_file" $@

# Enqueue tests example:
# ./launch_without_storage.sh None 2 5 false $(pwd)/src/kmeans.py -n 1024 -f 8 -d 2 -c 4
# ./launch_without_storage.sh None 2 15 false $(pwd)/src/kmeans.py -n 48000 -f 48 -d 20 -c 4
# ./launch_without_storage.sh None 3 60 true $(pwd)/src/kmeans.py -n 249999360 -f 1536 -d 100 -c 500 -i 5


# OUTPUTS:
# - compss-XX.out : Job output file
# - compss-XX.err : Job error file
# - ~/.COMPSs/JOB_ID/ : COMPSs files
