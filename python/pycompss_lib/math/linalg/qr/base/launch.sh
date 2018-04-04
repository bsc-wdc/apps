#!/bin/bash

  scriptDir=$(pwd)/$(dirname $0)
  EXEC_FILE=${scriptDir}/src/qr.py
  WORK_DIR=${scriptDir}/results/
  LOCAL_CLASSPATH=${scriptDir}/src/

  if [ ! -d $WORK_DIR ]; then
    mkdir $WORK_DIR
  fi

  export ComputingUnits=$8

  enqueue_compss \
    --job_dependency=$1 \
    --exec_time=$3 \
    --num_nodes=$2 \
    --tasks_per_node=$4 \
    --master_working_dir=$WORK_DIR \
    --worker_working_dir=scratch \
    --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
    --tracing=$5 \
    --graph=$5 \
    --lang=python \
    --debug \
    --pythonpath=$LOCAL_CLASSPATH \
    $EXEC_FILE $6 $7 $9 ${10}
    #--node_memory=50000 \
    #--debug

    # Params: Job_Dependency num_nodes exec_time tasks_per_node tracing MSIZE BSIZE ComputingUnits MKL_NUM_THREADS verify_result
    # Example: ./launch.sh None 2 10 16 true 4 512 8 16 False
