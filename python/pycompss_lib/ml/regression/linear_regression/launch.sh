#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=5
  tasksPerNode=16
  tracing=false
  
  # Set arguments:
  # Parameters: NumberOfPoints Fragments plotResult
  appArgs="25600000 64 False"

  # Execute specifc version launch
  # ${scriptDir}/1_base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
  # ${scriptDir}/2_noWaitOns/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
  # ${scriptDir}/3_initParallel/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
  ${scriptDir}/4_apps_objects/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
