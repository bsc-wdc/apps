#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=5
  tasksPerNode=16
  tracing=false

  # Set arguments
  appArgs="1000 3 3"

  # Execute specific version launch
  cd base
  ./run_local.sh $tracing $appArgs
  cd ..
