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
  dataFile=../../../../../../datasets/dbscan/data/5

  # Execute specific version launch
  cd base
  ./launch_local.sh $tracing $epsilon $minPoints $dataFile
  cd ..
