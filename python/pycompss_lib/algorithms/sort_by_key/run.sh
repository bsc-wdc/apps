#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)/$(dirname $0)

  # Set common arguments
  num_records=1000
  unique_keys=100
  key_length=5
  unique_values=100
  value_length=5
  num_partitions=2
  random_seed=8
  storage_location="dataset.txt"
  hash_function="False"
  tracing=false

  ${scriptDir}/generator/./generate_dataset.sh ${num_records} ${unique_keys} ${key_length} ${unique_values} ${value_length} ${num_partitions} ${random_seed} ${storage_location} ${hash_function}

  # Set arguments
  # appArgs="${storage_location}"

  # Execute specific version launch
  # 1_base/run.sh $tracing $appArgs
  # 2_hash/run.sh $tracing $appArgs
  # 3_files/run.sh $tracing $appArgs
  # 4_sortNumpy/run.sh $tracing $appArgs
  # 5_sortTotal/run.sh $tracing $appArgs

  # Set arguments
  appArgs="${num_records} ${unique_keys} ${key_length} ${unique_values} ${value_length} ${num_partitions} ${random_seed} True ${storage_location}"

  6_sortByKey/run.sh $tracing $appArgs
