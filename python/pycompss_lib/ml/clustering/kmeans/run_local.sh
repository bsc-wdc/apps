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
  appArgs="16000 3 4 16"
  #  numV = Number of points
  #  dim = Number of dimensions
  #  k = Number of centers
  #  numFrag = Number of fragments

  # Execute specific version launch
  cd latest
  ./run_local.sh $tracing $appArgs
  cd ..
