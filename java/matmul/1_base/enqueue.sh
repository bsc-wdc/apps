#!/bin/bash -e

workingDir=shared_disk
numNodes=2
execTime=60
appdir=$PWD
seed=1
msize=4
bsize=2

enqueue_compss -d --worker_working_dir=$workingDir --num_nodes=$numNodes --log_level=debug --exec_time=$execTime --appdir=$PWD matmul.randomGen.objects.Matmul $msize $bsize $seed

