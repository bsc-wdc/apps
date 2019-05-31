#!/bin/bash -e

workingDir=gpfs
numNodes=2
execTime=60
appdir=$PWD

enqueue_compss -d --worker_working_dir=$workingDir --num_nodes=$numNodes --log_level=debug --exec_time=$execTime --appdir=$PWD base/src/gen.py 10 10 20 10

