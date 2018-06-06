#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=30
  tasksPerNode=16
  tracing=false
  
  # Set arguments
  appArgs="100 100 200 10"

  # Execute specifcversion launch  
  ${scriptDir}/base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
