#!/bin/bash -e

  # THIS MUST BE INCLUDED INTO .bashrc
  echo "PLEASE, MAKE SURE THAT THE FOLLOWING LINES ARE IN YOUR .bashrc"
  echo "export COMPSS_PYTHON_VERSION=3-ML"
  echo "module use /apps/modules/modulefiles/tools/COMPSs/.custom"
  echo "module load TrunkJCB"
  # echo "module load COMPSs/Trunk"
  echo "module load hecuba/0.1.3_ML"

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
  module load hecuba/0.1.3_ML

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
  log_level="off"
  qos_flag="--qos=debug"
  workers_flag=""
  constraints=""

  # Create workers sandbox
  # mkdir -p "${WORK_DIR}/COMPSs_Sandbox"
  # --job_execution_dir="${WORK_DIR}" \
  # --worker_working_dir="${WORK_DIR}/COMPSs_Sandbox" \

  CPUS_PER_NODE=48
  WORKER_IN_MASTER=0

  shift 5

  # Those are evaluated at submit time, not at start time...
  COMPSS_VERSION=`ml whatis COMPSs 2>&1 >/dev/null | awk '{print $1 ; exit}'`
  HECUBA_VERSION=`ml whatis HECUBA 2>&1 >/dev/null | awk '{print $1 ; exit}'`

  # Enqueue job
  enqueue_compss \
    --job_name=wordcountOO_PyCOMPSs_Hecuba \
    --job_dependency="${job_dependency}" \
    --exec_time="${execution_time}" \
    --num_nodes="${num_nodes}" \
    \
    --cpus_per_node="${CPUS_PER_NODE}" \
    --worker_in_master_cpus="${WORKER_IN_MASTER}" \
    --scheduler=es.bsc.compss.scheduler.fifodata.FIFODataScheduler \
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
    --classpath=$HECUBA_ROOT/compss/ITF/StorageItf-1.0-jar-with-dependencies.jar:${APP_CLASSPATH}:${CLASSPATH} \
    --pythonpath=${APP_PYTHONPATH}:${PYTHONPATH} \
    --storage_props=$(pwd)/hecuba_confs/storage_props.cfg \
    --storage_home=$HECUBA_ROOT/compss/ \
    \
    --lang=python \
    \
    "$exec_file" $@ --use_storage


# Enqueue tests example:
# ./launch_with_Hecuba.sh None 2 5 false $(pwd)/src/wordcount.py -d $(pwd)/dataset

# OUTPUTS:
# - compss-XX.out : Job output file
# - compss-XX.err : Job error file
# - ~/.COMPSs/JOB_ID/ : COMPSs files
