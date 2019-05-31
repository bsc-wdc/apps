#!/bin/bash -e

workingDir=gpfs
numNodes=2
execTime=30
appdir=$PWD
numFragments=4
APPS_BASE=$PWD/../../..
enqueue_compss --classpath=$PWD/target/blastallone.jar --worker_working_dir=$workingDir --num_nodes=$numNodes --log_level=debug --exec_time=$execTime blast.Blast true $APPS_BASE/java/blast/deps/binaries/blastall $APPS_BASE/datasets/Blast/databases/swissprot/swissprot $APPS_BASE/datasets/Blast/sequences/sargasso_test.fasta $numFragments /tmp/ /tmp/result.txt


