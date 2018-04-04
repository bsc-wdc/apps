#!/usr/bin/env bash

if [ "$#" -gt 1 ]; then
    tracing=$1
    level=$2
    num_nodes=$3
    exe_time=$4
    shift 4
else
   echo "Usage: run tracing log_level num_nodes exe_time -e 5 -d 2 -a 2 -g 0.01 -c 10000 -m 1 -k rbf -t /.statelite/tmpfs/gpfs/home/bsc19/bsc19277/fenrir/compss_ML/data/3features_20k.dat"
   exit -1
fi

# Init sandboxed workspace and exe file

path="$(pwd)/../bin/"
pythonpath="$(pwd)/../"
exe="runner"

suffix=0
workspace="$(pwd)/target/nord3SVM_${num_nodes}_"

while [ -d "${workspace}${suffix}" ]; do
    ((++suffix))
done

mkdir -p "${workspace}${suffix}"
DEST="${workspace}${suffix}"

cd $DEST 

# Init output & error

outfile="${exe}.out"
errfile="${exe}.err"
reportfile="launch.cmd"


echo "
enqueue_compss \
 --tracing=$tracing \
 --log_level=$level \
 --lang=python \
 --num_nodes=$num_nodes \
 --exec_time=$exe_time \
 --classpath=${path} \
 --pythonpath=${pythonpath} \
 '${path}${exe}' $@ > >(tee ${outfile}) 2> >(tee ${errfile} >&2) " >> $reportfile

enqueue_compss \
 --tracing=$tracing \
 --log_level=$level \
 --lang=python \
 --num_nodes=$num_nodes \
 --exec_time=$exe_time \
 --classpath=${path} \
 --pythonpath=${pythonpath} \
 "${path}${exe}" $@ > >(tee ${outfile}) 2> >(tee ${errfile} >&2)
