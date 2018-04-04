#!/usr/bin/env bash

if [ "$#" -gt 1 ]; then
    tracing=$1
    level=$2
    graph=$3
    shift 3

else
   echo "Usage: run tracing log_level graph [experiment arity depth gamma C max_iter kernel [datasize for exp 5]]"
   echo "./local-svm_runner.sh true debug false -e 5 -d 2 -a 2 -g 0.01 -c 10000 -m 1 -k rbf"
   exit -1
fi

# Init sandboxed workspace and exe file

path="$(pwd)/../bin/"
pythonpath="$(pwd)/../"
exe="runner"

suffix=0
workspace="$(pwd)/target/localSVM_"

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


echo "runcompss \
 --tracing=$tracing \
 --log_level=$level \
 --graph=$graph \
 --lang=python \
 --pythonpath=${path} \
 '${path}${exe}'  $@"  > >(tee ${reportfile})


runcompss \
 --tracing=$tracing \
 --log_level=$level \
 --graph=$graph \
 --lang=python \
 --pythonpath=${pythonpath} \
 "${path}${exe}" $@
