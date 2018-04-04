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
  appArgs="10 100"

  # Execute specifcversion launch
  ${scriptDir}/2_base_oo/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs

#  numFragments = number of fragments
#  numEntries = number of (k, v) pairs withien each fragment
