#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  tracing=false
  
  # Set arguments
  appArgs="100"

  # Execute specifcversion launch  
  ${scriptDir}/base/run_local.sh $tracing $appArgs
