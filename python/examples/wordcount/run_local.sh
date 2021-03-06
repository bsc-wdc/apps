#!/bin/bash -e

  # Define script directory for relative calls
  # scriptDir=$(dirname $0)
  # scriptDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  scriptDir=$(pwd)

  # Generate test dataset
  dataset="${scriptDir}/dataset_4f_1mb/"
  ${scriptDir}/generator/generate_dataset.sh 4 1 ${dataset}

  # Set common arguments
  tracing=false

  # Set arguments
  appArgs="${dataset} True"

  # Execute specific version launch
  ${scriptDir}/wordcount/run_local.sh $tracing $appArgs


  ######################################################
  # APPLICATION EXECUTION EXAMPLE
  # Call:
  #       ./run_local.sh
  #
  # Example:
  #       ./run_local.sh
  #
