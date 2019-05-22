#!/bin/bash -e

  # WARNING ========================> Needs a Python version with TF
  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=5
  executionTime=120
  tasksPerNode=48
  tracing=false
  
  # Set arguments
  appArgs=". 2"

  # Execute specifcversion launch  
  ${scriptDir}/base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
