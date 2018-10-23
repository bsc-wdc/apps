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
  appArgs="16000 3 4 16"
  #  numV = Number of points
  #  dim = Number of dimensions
  #  k = Number of centers
  #  numFrag = Number of fragments

  # Execute specific version launch
  ${scriptDir}/latest/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
