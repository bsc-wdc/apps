#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=10
  tasksPerNode=32
  tracing=true

  # Set arguments
  matrixSize=4
  blockSize=8192
  computingUnits=8
  MKL_NUM_THREADS=16

  # Execute specifcversion launch  
  ${scriptDir}/2_init_parallel/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $matrixSize $blockSize $computingUnits $MKL_NUM_THREADS

