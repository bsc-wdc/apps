#!/bin/bash

  #Define script directory for relative calls
  scriptDir=$(dirname $0)
  EXEC_FILE=${scriptDir}/src/matmul.py 
  LOCAL_CLASSPATH=${scriptDir}/src/

  enqueue_compss \
    --job_dependency=$1 \
    --exec_time=$3 \
    --num_nodes=$2 \
    --tasks_per_node=$4 \
    --master_working_dir=. \
    --worker_working_dir=scratch \
    --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
    --tracing=$5 \
    --lang=python \
    --pythonpath=$LOCAL_CLASSPATH \
    $EXEC_FILE $6 $7

  # Params: Job_Dependency num_nodes exec_time tasks_per_node tracing MSIZE BSIZE 
  # Example: ./launch.sh None 2 10 16 true 4 128
