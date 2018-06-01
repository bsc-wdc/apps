#!/bin/bash -e

# $1: <Integer> Number of entries
# $2: <Integer> Unique keys
# $3: <Integer> Key length
# $4: <Integer> Unique values
# $5: <Integer> Value length
# $6: <Integer> Number of partitions
# $7: <Integer> Random seed
# $8: <String>  Output folder
# $9: <Boolean> Hash function

# Define script directory for relative calls
scriptDir=$(dirname $0)

python ${scriptDir}/src/generator.py $1 $2 $3 $4 $5 $6 $7 $8 $9
