#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=15
  tasksPerNode=16
  tracing=false
  
  # Set arguments
  appArgs="1024 /gpfs/projects/bsc19/COMPSs_DATASETS/neurons/spikes.dat"

  # Execute specifcversion launch  
  ${scriptDir}/notInout/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs


