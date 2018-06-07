#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)

  # Set common arguments
  nums=102400
  max_num=200000
  dataset="dataset.txt"
  tracing=false

  ${scriptDir}/generator/generate_dataset.sh ${nums} ${max_num} ${dataset}
  
  # Set arguments
  appArgs="${scriptDir}/${dataset} 5 600"

  # Execute specific version launch
  cd base
  ./run_local.sh $tracing $appArgs
  cd ..
