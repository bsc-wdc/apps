#!/bin/bash -e

workingDir=gpfs
numNodes=2
execTime=60
appdir=$PWD
numClusters=100
numIterations=10
numDimensions=100
numPoints=99840
numFragments=512
seed=5
scaleFactor=10
sameFragments=false

enqueue_compss -d --worker_working_dir=$workingDir --num_nodes=$numNodes --log_level=debug --exec_time=$execTime --appdir=$PWD kmeans_frag.KMeans_frag -c $numClusters -i $numIterations -n $numPoints -d $numDimensions -f $numFragments -s $seed -r $scaleFactor -ef $sameFragments

