#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  tracing=false
  
  # Set arguments
  appArgs="16 4096"

  # Execute specific version launch
  ${scriptDir}/base/run_local.sh $tracing $appArgs
