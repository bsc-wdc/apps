#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=5
  tasksPerNode=16
  tracing=false

  # Set arguments:
  # Parameters: NumberOfPoints Fragments plotResult
  appArgs="1024 8 False"

  # Execute specific version launch
  # cd 1_base
  # cd 2_noWaitOns
  # cd 3_initParallel
  cd 4_apps_objects
  ./run_local.sh $tracing $appArgs
  cd ..
