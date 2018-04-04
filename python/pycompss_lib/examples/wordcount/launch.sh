#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=10
  tasksPerNode=16
  tracing=true
  
  # Set arguments
  appArgs="/gpfs/projects/bsc19/COMPSs_DATASETS/wordcount/data/dataset_64f_16mb"

  # Execute specifcversion launch  
  ${scriptDir}/6_tutorial/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs
