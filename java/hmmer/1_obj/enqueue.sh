#!/bin/bash -e

workingDir=gpfs
numNodes=2
execTime=60
appdir=$PWD
APPS_BASE=$PWD/../../..
export HMMER_BINARY=$APPS_BASE/java/hmmer/deps/binaries/hmmpfam
enqueue_compss -d --worker_working_dir=$workingDir --worker_in_master_cpus=0 --num_nodes=$numNodes --log_level=debug --exec_time=$execTime --appdir=$PWD --classpath=$APPS_BASE/java/hmmer/1_obj/target/hmmerobj.jar hmmerobj.HMMPfam $APPS_BASE/datasets/Hmmer/smart.HMMs.bin $APPS_BASE/datasets/Hmmer/512seq /tmp/hmmer.result 4 4


