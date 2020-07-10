#!/bin/bash -e

  module load gcc/8.1.0
  export COMPSS_PYTHON_VERSION=3-ML
  module use /apps/modules/modulefiles/tools/COMPSs/.custom
  module load TrunkJCB
  # module load COMPSs/2.6.3
  module load mkl/2018.1
  module load impi/2018.1
  module load opencv/4.1.2
  module load DATACLAY/2.4.dev

  # Retrieve script arguments
  job_dependency=${1:-None}
  num_nodes=${2:-2}
  execution_time=${3:-5}
  tracing=${4:-false}
  exec_file=${5:-$(pwd)/src/kmeans.py}

  # Freeze storage_props into a temporal 
  # (allow submission of multiple executions with varying parameters)
  STORAGE_PROPS=`mktemp -p ~`
  cp $(pwd)/dataClay_confs/storage_props.cfg "${STORAGE_PROPS}"

  if [[ ! ${tracing} == "false" ]]
  then
    extra_tracing_flags="\
      --jvm_workers_opts=\"-javaagent:/apps/DATACLAY/dependencies/aspectjweaver.jar\" \
      --jvm_master_opts=\"-javaagent:/apps/DATACLAY/dependencies/aspectjweaver.jar\" \
"

    echo "Adding DATACLAYSRV_START_CMD to storage properties file"
    echo "\${STORAGE_PROPS}=${STORAGE_PROPS}"
    echo "" >> ${STORAGE_PROPS}
    echo "DATACLAYSRV_START_CMD=\"--tracing\"" >> ${STORAGE_PROPS}
  fi

  # Define script variables
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR=${SCRIPT_DIR}/
  APP_CLASSPATH=${SCRIPT_DIR}/src
  APP_PYTHONPATH=${SCRIPT_DIR}/src

  # Freeze storage_props into a temporal 
  # (allow submission of multiple executions with varying parameters)
  STORAGE_PROPS=`mktemp -p ~`
  cp $(pwd)/dataClay_confs/storage_props.cfg "${STORAGE_PROPS}"

  if [[ ! ${tracing} == "false" ]]
  then
    extra_tracing_flags="\
      --jvm_workers_opts=\"-javaagent:/apps/DATACLAY/dependencies/aspectjweaver.jar\" \
      --jvm_master_opts=\"-javaagent:/apps/DATACLAY/dependencies/aspectjweaver.jar\" \
"

    echo "Adding DATACLAYSRV_START_CMD to storage properties file"
    echo "\${STORAGE_PROPS}=${STORAGE_PROPS}"
    echo "" >> ${STORAGE_PROPS}
    echo "DATACLAYSRV_START_CMD=\"--tracing\"" >> ${STORAGE_PROPS}
  fi

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
  COMPSS_VERSION=`module load whatis COMPSs 2>&1 >/dev/null | awk '{print $1 ; exit}'`
  DATACLAY_VERSION=`module load whatis DATACLAY 2>&1 >/dev/null | awk '{print $1 ; exit}'`

  # Enqueue job
  enqueue_compss \
    --job_name=kmeans_PyCOMPSs_dataClay \
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
    --classpath=${DATACLAY_JAR} \
    --pythonpath=${APP_PYTHONPATH}:${PYCLAY_PATH}:${PYTHONPATH} \
    --storage_props=${STORAGE_PROPS} \
    --storage_home=$COMPSS_STORAGE_HOME \
    --prolog="$DATACLAY_HOME/bin/dataclayprepare,$(pwd)/src/storage_model/,$(pwd)/src/,storage_model,python" \
    \
    ${extra_tracing_flags} \
    \
    --lang=python \
    \
    "$exec_file" $@ --use_storage

# Enqueue tests example:
# ./launch_with_dataClay.sh None 2 10 false $(pwd)/src/kmeans.py -n 1024 -f 8 -d 2 -c 4
# ./launch_with_dataClay.sh None 3 60 true $(pwd)/src/kmeans.py -n 249999360 -f 1536 -d 100 -c 500 -i 5

# OUTPUTS:
# - compss-XX.out : Job output file
# - compss-XX.err : Job error file
# - ~/.COMPSs/JOB_ID/ : COMPSs files
