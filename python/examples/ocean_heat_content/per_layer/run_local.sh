#!/bin/bash -e

# Define script variables
scriptDir=$(pwd)/$(dirname $0)
execFile=${scriptDir}/oceanheatcontent.py
appClasspath=${scriptDir}/
appPythonpath=${scriptDir}/

runcompss \
--classpath=$appClasspath \
--pythonpath=$appPythonpath \
--log_level=off \
-g \
-t \
--summary \
$execFile $@

###########################
# Usage example:
#
# ./run_local.sh /path/to/oceanheatcontent/dataset/ \
#                /path/to/mesh_mask.nc \
#                /path/to/mask_regions.nc \
#                /path/to/results/
###########################
