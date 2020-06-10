#!/bin/bash -e

  # Define script constants
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  
  # Load modules
  module purge
  module load intel/2017.4
  module load mkl/2017.4
  export COMPSS_PYTHON_VERSION=3
  module use /apps/modules/modulefiles/tools/COMPSs/.custom
  module load TrunkNested

  # COMPSs configuration
  # SCHEDULER="es.bsc.compss.scheduler.fifodata.FIFODataScheduler"
  # SCHEDULER="es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler"
  # SCHEDULER="es.bsc.compss.scheduler.multiobjective.MOScheduler"
  num_workers=2
  num_nodes=$(( num_workers + 1 ))

  # Application configuration
  num_estimators=3840
  num_test_estimators=$(( num_workers * 48 * 2))
  length=$(( 60 + 4 * (( (num_estimators * 8) / (num_workers * 48 - 1) )) ))
  length=$(( (length / 60) + 1 ))
  # length=20
  app_exec="randomforest.RandomForest"

  # Create output directory
  mkdir -p "${SCRIPT_DIR}/../output"

  # Run job
  enqueue_compss \
    --qos=debug \
    --num_nodes="${num_nodes}" \
    --exec_time="${length}" \
    \
    --cpus_per_nodes=48 \
    --node_memory=50000 \
    --worker_in_master_cpus=1 \
    \
    --master_working_dir="${SCRIPT_DIR}"/../output/ \
    --worker_working_dir= \
    --base_log_dir="${SCRIPT_DIR}"/../output \
    --classpath="${SCRIPT_DIR}"/../target/random_forest.jar \
    \
    --log_level=debug \
    --jvm_workers_opts="-Dcompss.worker.removeWD=false" \
    \
    --agents=plain \
    --method_name="main" \
    --array \
    "${app_exec}" 30000 40 200 20 2 1 2 "true" 0 "${num_estimators}" "${num_test_estimators}"

