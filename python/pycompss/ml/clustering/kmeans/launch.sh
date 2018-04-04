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

  # Execute specifcversion launch  
  ${scriptDir}/12_latest/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs

#  numV = numero de punts
#  dim = dimensions del punt
#  k = numero de centres
#  numFrag = numero de fragments 
