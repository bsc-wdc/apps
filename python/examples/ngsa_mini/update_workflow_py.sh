#!/bin/bash

if [ $# -lt "1" ]; then
  $1="1_compssify"
fi


scp $1/src/workflow.py bsc19509@mn1.bsc.es:/gpfs/projects/bsc19/COMPSs_APPS/ngsa_mini/ngsa-mini-py/bin/
