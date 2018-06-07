#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)

  # Set common arguments
  tracing=false
  
  # Set arguments
  appArgs=". 2"


  # Execute specific version launch
  cd base
  ./run_local.sh $tracing $appArgs
  cd ..
