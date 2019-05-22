#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=5
  executionTime=30
  tracing=false

  # Set arguments
  appArgs="4 512 8 False"
  #  - MSIZE = 4 = Matrix size
  #  - BSIZE = 512 = Block size
  #  - mkl_threads = 8
  #  - Validate results = False

  # Execute specifcversion launch
  # ${scriptDir}/base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
  ${scriptDir}/base/launch.sh $jobDependency $numNodes $executionTime $tracing $appArgs
