#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=15
  tasksPerNode=16
  tracing=false
  
  # Set arguments
  appArgs="16 4096"

  # Execute specifc version launch
  ${scriptDir}/base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
