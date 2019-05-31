#!/bin/bash -e

workingDir=gpfs
numNodes=2
execTime=60
appdir=$PWD
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

enqueue_compss -d --worker_working_dir=$workingDir --num_nodes=$numNodes --log_level=debug --exec_time=$execTime --appdir=$PWD sortByKey/src/sort.py 10 5 3 5 2 5 12345 false undefined

