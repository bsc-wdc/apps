#!/usr/bin/env bash

if [ "$#" -eq 3 ]; then
    tracing=$1
    level=$2
    num_models=$3
else
   echo "Usage: run tracing log_level num_models"
   exit -1
fi

# Init sandboxed workspace and exe file

path="$(pwd)/../compssml/tf/"
pythonpath="$(pwd)/../"
exe="compss_mnist.py"

suffix=0
workspace="$(pwd)/target/local_tf_MNIST_"

while [ -d "${workspace}${suffix}" ]; do
    ((++suffix))
done

mkdir -p "${workspace}${suffix}"
mkdir -p "${workspace}${suffix}/models"
DEST="${workspace}${suffix}"
cd $DEST

# Init output & error

suffix=0
outfile="${exe}.out"
errfile="${exe}.err"
reportfile="launch.cmd"

echo " * Launching $exe with CMD:" >> $reportfile
echo "runcompss \
 --tracing=$tracing \
 --log_level=$level \
 --lang=python \
 --pythonpath=${pythonpath} \
 ${path}${exe} $DEST $num_models"  > >(tee ${reportfile})


runcompss \
 --tracing=$tracing \
 --log_level=$level \
 --lang=python \
 --pythonpath=${pythonpath} \
 ${path}${exe} $DEST $num_models > >(tee ${outfile}) 2> >(tee ${errfile} >&2)
