#!/bin/bash -e
  
  # Load module for COMPSs/Agents
  module use /apps/modules/modulefiles/tools/COMPSs/.custom
  module load TrunkAgents
#  SCHEDULER="es.bsc.compss.scheduler.fifodata.FIFODataScheduler"
#  SCHEDULER="es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler"
#  SCHEDULER="es.bsc.compss.scheduler.multiobjective.MOScheduler"
  NUM_ESTIMATORS=3840
  NUM_NODES=${1}
  NUM_TEST_ESTIMATORS=$(( NUM_NODES * 48 * 2))

  # Define script constants
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

  # Define application arguments
  app_exec="randomforest.RandomForest"

  LENGTH=$(( 60 + 4 * (( (NUM_ESTIMATORS * 8) / (NUM_NODES * 48 - 1) )) ))
  LENGTH=$(( (LENGTH / 60) + 1 ))
#  LENGTH=20
  
  # Run job
  enqueue_compss \
    --qos=debug \
    --num_nodes=$(( NUM_NODES + 1 )) \
    --worker_in_master_cpus=1 \
    --exec_time=${LENGTH} \
    \
    --classpath="/home/bsc19/bsc19111/agents_comparison/random_forest/application/target/random_forest.jar" \
    --jvm_workers_opts="-Dcompss.worker.removeWD=false" \
    \
    --agents=plain \
    \
    --method_name="generateRandomModelWithTest" \
    --array \
    "${app_exec}" 30000 40 200 20 2 1 2 "true" 0 ${NUM_ESTIMATORS} ${NUM_TEST_ESTIMATORS}

