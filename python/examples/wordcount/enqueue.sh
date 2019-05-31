#!/bin/bash -e

workingDir=gpfs
numNodes=2
execTime=60
appdir=$PWD

enqueue_compss -d --worker_working_dir=$workingDir --num_nodes=$numNodes --log_level=debug --exec_time=$execTime --appdir=$PWD --pythonpath=$PWD/wordcount/src wordcount/src/wordcount.py /gpfs/projects/bsc19/COMPSs_DATASETS/wordcount/data/dataset_64f_16mb/ True 
