#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)

  # Set common arguments
  tracing=false

  # Set arguments:
  # Parameters: NumberOfPoints Fragments plotResult
  appArgs="1024 8 False"

  # Execute specific version launch
  # cd base
  # cd noWaitOns
  # cd initParallel
  cd apps_objects
  ./run_local.sh $tracing $appArgs
  cd ..
