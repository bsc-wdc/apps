#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=5
  tasksPerNode=48
  tracing=false

  # Set arguments
  appArgs="4 512 8"
  #  - MSIZE = 4 = Matrix size
  #  - BSIZE = 512 = Block size
  #  - mkl_threads = 8

  # Execute specifcversion launch
  # ${scriptDir}/base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
  ${scriptDir}/base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
