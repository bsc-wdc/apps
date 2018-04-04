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
  appArgs="10 5 3 5 2 5 12345 false undefined"

  # Execute specifcversion launch  
  ${scriptDir}/6_sortByKey/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
