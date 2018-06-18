#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)

  # Set common arguments
  tracing=false

  # Set arguments:
  # Parameters: NumberOfPoints Fragments plotResult
  appArgs="0.1 10 5"
  epsilon=0.1
  minPoints=10
  dataFile=5

  # Execute specific version launch
  ./launch_local.sh $tracing $epsilon $minPoints $dataFile
