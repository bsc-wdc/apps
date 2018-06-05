#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)

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
  # cd 1_base
  # cd 2_hash
  # cd 3_files
  # cd 4_sortNumpy
  # cd 5_sortTotal
  # ./run_local.sh $tracing $appArgs
  # cd ..

  # Set arguments
  appArgs="${num_records} ${unique_keys} ${key_length} ${unique_values} ${value_length} ${num_partitions} ${random_seed} True ${storage_location}"

  cd 6_sortByKey
  ./run_local.sh $tracing $appArgs
  cd ..
