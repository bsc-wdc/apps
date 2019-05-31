#!/bin/bash -e

workingDir=gpfs
numNodes=2
execTime=60
export CLASSPATH=$CLASSPATH:$PWD/reverseindex.jar
export CLASSPATH=$CLASSPATH:$PWD/lib/htmlparser.jar

enqueue_compss -d --worker_working_dir=$workingDir --num_nodes=$numNodes --log_level=debug --exec_time=$execTime --appdir=$PWD reverse.Reverse true $PWD/test 3 $PWD/out.txt /tmp

