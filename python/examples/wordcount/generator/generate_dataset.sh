#!/bin/bash -e

# $1: Number of files
# $2: Size of each file in MBytes
# $3: Output dataset path

# Define script directory for relative calls
scriptDir=$(dirname $0)

python ${scriptDir}/src/generator.py $1 $2 $3
