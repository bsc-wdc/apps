#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)

  # Set common arguments
  tracing=false
  
  # Set arguments
  appArgs="16000 3 16"

  # Execute specific version launch
  cd base
  ./run_local.sh $tracing $appArgs
  cd ..
