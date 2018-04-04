#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=5
  tasksPerNode=16
  tracing=false

  # Set arguments
  matrixSize=4
  blockSize=4096
  computingUnits=4
  MKL_NUM_THREADS=16

  # Execute specifcversion launch  
  ${scriptDir}/1_base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $matrixSize $blockSize $computingUnits $MKL_NUM_THREADS False

