#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=5
  tasksPerNode=16
  tracing=false
  
  # Set arguments
  appArgs="16 3 0 1"

  # Execute specifcversion launch  
  ${scriptDir}/1_base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
