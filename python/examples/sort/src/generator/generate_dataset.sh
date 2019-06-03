#!/bin/bash -e

# $1: Numbers
# $2: Max number
# $3: Dataset output file

# Define script directory for relative calls
scriptDir=$(dirname $0)

python ${scriptDir}/src/generator.py $1 $2 $3