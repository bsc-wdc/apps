#!/bin/bash -e

  # Define script constants
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

  # Script params
  log_level=${1:-"debug"}
  job_dep=${2:-"None"}
  
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
  num_workers=1
  num_nodes=$(( num_workers + 1 ))
  exec_time=30

  # Application configuration
  num_simple_apps=1
  num_increments=1
  app_exec="simple.files.Simple"

  # Create output directory
  mkdir -p "${SCRIPT_DIR}/../../output"

  # Run job
  enqueue_compss \
    --job_dependency="${job_dep}" \
    --num_nodes="${num_nodes}" \
    --exec_time="${exec_time}" \
    \
    --cpus_per_nodes=48 \
    --node_memory=50000 \
    --worker_in_master_cpus=1 \
    \
    --master_working_dir="${SCRIPT_DIR}"/../../output/ \
    --worker_working_dir=/gpfs/scratch/bsc19/bsc19533 \
    --base_log_dir="${SCRIPT_DIR}"/../../output \
    --classpath="${SCRIPT_DIR}"/../target/simple_files.jar \
    \
    --log_level="${log_level}" \
    --jvm_workers_opts="-Dcompss.worker.removeWD=false" \
    \
    --agents=plain \
    --method_name="main" \
    --array \
    "${app_exec}" "${num_simple_apps}" "${num_increments}"

 # --qos=debug --tracing=true --graph=true --summary
