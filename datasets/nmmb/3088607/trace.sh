#!/bin/bash

source /gpfs/apps/MN3/COMPSs/TrunkJavi//Dependencies/extrae-openmpi/etc/extrae.sh

export EXTRAE_CONFIG_FILE=/gpfs/home/bsc19/bsc19234/extrae_mine.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so # For C apps
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so # For Fortran apps

## Run the desired program
$*

