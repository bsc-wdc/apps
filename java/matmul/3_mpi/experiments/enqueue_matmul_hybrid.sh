#!/bin/bash

  MSIZE=$1
  BSIZE=$2
  NUM_PROCS=$3
  NUM_NODES=$4
  EXEC_TIME=$5
  JOB_DEP=$6

  scriptDir=$(pwd)
  binary=${scriptDir}/bin/matmul
  TYPE=2

  export MATMUL_BINARY=${binary}
  export MPI_PROCS=1
  export CUS=$NUM_PROCS
  export BSIZE=$BSIZE

  echo "Enqueueing job for TYPE=$TYPE ; MSIZE=$MSIZE ; BSIZE=$BSIZE ; NUM_PROCS=$NUM_PROCS"
  enqueue_compss \
    --exec_time=$EXEC_TIME \
    --num_nodes=$NUM_NODES \
    --tasks_per_node=16 \
    --job_execution_dir=. \
    --worker_working_dir=scratch \
    --classpath=${scriptDir}/bin/matmul.jar \
    --network=infiniband \
    --log_level=off \
    --tracing=false \
    --graph=false \
    --summary \
    --job_dependency=$JOB_DEP \
    matmul.files.Matmul $TYPE $MSIZE $BSIZE

