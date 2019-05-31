#!/bin/bash -e

workingDir=gpfs
numNodes=2
execTime=60
appdir=$PWD
nums=102400
  max_num=200000
  dataset="dataset.txt"
  tracing=false



enqueue_compss -d --worker_working_dir=$workingDir --num_nodes=$numNodes --log_level=debug --exec_time=$execTime --appdir=$PWD base/src/sort.py /gpfs/projects/bsc19/COMPSs_DATASETS/sortNumbers/Random6000.txt ${nums} ${max_num}

