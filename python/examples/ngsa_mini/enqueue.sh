#!/bin/bash -e

workingDir=gpfs
numNodes=2
execTime=60
appdir=$PWD


A_MINI_HOME="/gpfs/projects/bsc19/COMPSs_APPS/ngsa_mini/ngsa-mini-py"
INPUT_PATH="/gpfs/projects/bsc19/COMPSs_DATASETS/ngsa_mini"

enqueue_compss -d --worker_working_dir=$workingDir --num_nodes=$numNodes --log_level=debug --exec_time=$execTime --appdir=$PWD $PWD/base/src/workflow.py $INPUT_PATH/bwa_db/reference.fa $INPUT_PATH/seq_contig.md $INPUT_PATH/reference.fa $INPUT_PATH/reference.fa.fai $INPUT_PATH/work/wfinput_08_8000000 8
