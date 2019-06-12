#!/bin/bash -e

# Define script variables
scriptDir=$(pwd)/$(dirname $0)
execFile=${scriptDir}/oceanheatcontent.py
appClasspath=${scriptDir}/
appPythonpath=${scriptDir}/

nodes=$1
walltime=$2
shift 2

enqueue_compss \
--num_nodes=$nodes \
--exec_time=$walltime \
--gpus_per_node=4 \
--classpath=$appClasspath \
--pythonpath=$appPythonpath \
--log_level=off \
--qos=bsc_cs \
-g \
-t \
--summary \
$execFile $@

###########################
# Usage example:
#
# ./launch.sh 2 20 \
#             /gpfs/projects/bsc19/COMPSs_DATASETS/oceanheatcontent/all/ \
#             /gpfs/projects/bsc32/cs_collaboration/numba_ohc/dataset/mesh_mask_nemo.Ec3.2_O1L75.nc \
#             /gpfs/projects/bsc32/cs_collaboration/numba_ohc/dataset/mask.regions.Ec3.2_O1L75.nc \
#             /gpfs/projects/bsc32/cs_collaboration/numba_ohc/results/ \
#             True
###########################
