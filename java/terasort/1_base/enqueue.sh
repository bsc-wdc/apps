#!/bin/bash -e

workingDir=shared_disk
numNodes=2
execTime=60
appdir=$PWD

enqueue_compss -d --worker_working_dir=$workingDir --num_nodes=$numNodes --log_level=debug --exec_time=$execTime --appdir=$PWD terasort.random.Sort 4 10 5 4 10 2 1 

