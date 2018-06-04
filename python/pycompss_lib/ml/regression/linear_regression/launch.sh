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
  # Parameters: NumberOfPoints Fragments
  appArgs="25600000 64"

  # Execute specifcversion launch  
  ${scriptDir}/3_initParallel/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
