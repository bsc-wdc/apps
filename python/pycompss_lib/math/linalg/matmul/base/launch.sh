#!/bin/bash

  #Define script directory for relative calls
  scriptDir=$(pwd)
  EXEC_FILE=${scriptDir}/src/matmul.py 
  LOCAL_CLASSPATH=${scriptDir}/src/

  export ComputingUnits="1"

  enqueue_compss \
    --job_dependency=$1 \
    --exec_time=$3 \
    --num_nodes=$2 \
    --cpus_per_node=$4 \
    --master_working_dir=. \
    --worker_working_dir=scratch \
    --scheduler="es.bsc.compss.scheduler.fifoDataScheduler.FIFODataScheduler" \
    --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
    --tracing=$5 \
    --debug \
    --lang=python \
    --pythonpath=$LOCAL_CLASSPATH \
    --qos=debug \
    $EXEC_FILE $6 $7 $8

  # Params: Job_Dependency num_nodes exec_time tasks_per_node tracing MSIZE BSIZE MKL_Threads
  # Example: ./launch.sh None 2 10 16 true 4 128 1
