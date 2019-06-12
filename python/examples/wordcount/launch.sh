#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=10
  tracing=true

  # Set arguments
  appArgs="/gpfs/projects/bsc19/COMPSs_DATASETS/wordcount/data/dataset_64f_16mb True"

  # Execute specific version launch
  ${scriptDir}/wordcount/launch.sh $jobDependency $numNodes $executionTime $tracing $appArgs


  ######################################################
  # APPLICATION EXECUTION EXAMPLE
  # Call:
  #       ./launch.sh
  #
  # Example:
  #       ./launch.sh
  #
