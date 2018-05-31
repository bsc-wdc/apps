#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=5
  tracing=false
  
  # Set arguments
  appArgs="/gpfs/projects/bsc19/COMPSs_DATASETS/sortNumbers/Random6000.txt 5 600"

  # Execute specific version launch
  ${scriptDir}/1_base/launch.sh $jobDependency $numNodes $executionTime $tracing $appArgs
