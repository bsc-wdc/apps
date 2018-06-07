#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  tracing=false
  
  # Set arguments
  appArgs="16 3 0 1"

  # Execute specifcversion launch  
  ${scriptDir}/base/run_local.sh $tracing $appArgs

