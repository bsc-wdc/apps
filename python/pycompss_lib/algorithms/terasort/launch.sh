#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=5
  tasksPerNode=48
  tracing=false

  # Set arguments
  appArgs="10 100"
  #  - numFragments =  10 = number of fragments
  #  - numEntries   = 100 = number of (k, v) pairs within each fragment

  # Execute specifcversion launch
  # ${scriptDir}/1_base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
  ${scriptDir}/2_base_oo/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
