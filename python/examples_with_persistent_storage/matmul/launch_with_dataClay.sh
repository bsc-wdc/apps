#!/bin/bash -e

  # THIS MUST BE INCLUDED INTO .bashrc
  echo "PLEASE, MAKE SURE THAT THE FOLLOWING LINES ARE IN YOUR .bashrc"
  echo "module load gcc/8.1.0"
  echo "export COMPSS_PYTHON_VERSION=3-ML"
  echo "module use /apps/modules/modulefiles/tools/COMPSs/.custom"
  echo "module load TrunkJCB"
  # echo "module load COMPSs/2.6.3"
  echo "module load mkl/2018.1"
  echo "module load impi/2018.1"
  echo "module load opencv/4.1.2"
  echo "module load python/3.6.4_ML"
  echo "module load DATACLAY/2.0rc"

  # read -p "Continue? (y|n) " -n 1 -r
  # echo
  # if [[ ! $REPLY =~ ^[Yy]$ ]]
  # then
  #     [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
  # fi

  module load gcc/8.1.0
  export COMPSS_PYTHON_VERSION=3-ML
  module use /apps/modules/modulefiles/tools/COMPSs/.custom
  module load TrunkJCB
  # module load COMPSs/2.6.3
  module load mkl/2018.1
  module load impi/2018.1
  module load opencv/4.1.2
  module load python/3.6.4_ML
  module load DATACLAY/2.0rc

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
    --classpath=$DATACLAY_JAR:$DATACLAY_DEPENDENCY_LIBS/*:${APP_CLASSPATH}:${CLASSPATH} \
    --pythonpath=${APP_PYTHONPATH}:${PYTHONPATH} \
    --storage_props=$(pwd)/dataClay_confs/storage_props.cfg \
    --storage_home=$COMPSS_STORAGE_HOME \
    --prolog=$(pwd)/dataClay_confs/register.sh \
    \
    --lang=python \
    \
    "$exec_file" $@ --use_storage

# Enqueue tests example:
# ./launch_with_dataClay.sh None 2 5 false $(pwd)/src/matmul.py -b 4 -e 4 --check_result

# OUTPUTS:
# - compss-XX.out : Job output file
# - compss-XX.err : Job error file
# - ~/.COMPSs/JOB_ID/ : COMPSs files
