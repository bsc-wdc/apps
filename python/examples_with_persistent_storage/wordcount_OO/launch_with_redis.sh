#!/bin/bash -e

  # THIS MUST BE INCLUDED INTO .bashrc
  echo "PLEASE, MAKE SURE THAT THE FOLLOWING LINE IS IN YOUR .bashrc"
  echo "export PATH=/apps/COMPSs/Storage/Redis/bin:\$PATH"

  # read -p "Continue? (y|n) " -n 1 -r
  # echo
  # if [[ ! $REPLY =~ ^[Yy]$ ]]
  # then
  #     [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
  # fi

  export COMPSS_PYTHON_VERSION=3-ML
  module use /apps/modules/modulefiles/tools/COMPSs/.custom
  module load TrunkJCB
  # module load COMPSs/Trunk

  module load ruby
  export PATH=/apps/COMPSs/Storage/Redis/bin:$PATH

  # Not working - requires to be included into .bashrc?
  # module use /apps/modules/modulefiles/tools/COMPSs/.custom
  # module load Redis

  # Storage-related paths
  # Change these paths if you want to use other storage implementations
  STORAGE_HOME=${COMPSS_HOME}/Tools/storage/redis
  STORAGE_CLASSPATH=${STORAGE_HOME}/compss-redisPSCO.jar

  # Retrieve script arguments
  job_dependency=${1:-None}
  num_nodes=${2:-2}
  execution_time=${3:-5}
  tracing=${4:-false}
  exec_file=${5:-$(pwd)/src/wordcount.py}

  # Define script variables
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR=${SCRIPT_DIR}/
  APP_CLASSPATH=${SCRIPT_DIR}/src
  APP_PYTHONPATH=${SCRIPT_DIR}/src

  # Define application variables
  graph=$tracing
  log_level="debug"
  qos_flag="--qos=off"
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
    --job_name=wordcountOO_PyCOMPSs_redis \
    --job_dependency="${job_dependency}" \
    --exec_time="${execution_time}" \
    --num_nodes="${num_nodes}" \
    \
    --cpus_per_node="${CPUS_PER_NODE}" \
    --worker_in_master_cpus="${WORKER_IN_MASTER}" \
    --scheduler=es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler \
    \
    "${workers_flag}" \
    \
    --worker_working_dir=/gpfs/scratch/bsc19/bsc19234/ \
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
    "$exec_file" $@ --use_storage

# Enqueue tests example:
# ./launch_with_redis.sh None 2 5 false $(pwd)/src/wordcount.py -d $(pwd)/dataset

# OUTPUTS:
# - compss-XX.out : Job output file
# - compss-XX.err : Job error file
# - ~/.COMPSs/JOB_ID/ : COMPSs files
