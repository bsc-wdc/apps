#!/bin/bash
numNodes=$1
numMutation=$2
export MDCU=48
export COMPSS_PYTHON_VERSION=3
module load COMPSs/Trunk
module load singularity/3.5.2
JAVA_TOOL_OPTIONS=-Xss1280k
enqueue_compss --pythonpath=$PWD/src -t -g -d --python_interpreter=python3 --qos=debug --num_nodes=$numNodes --worker_in_master_cpus=48 --worker_working_dir=shared_disk \
	$PWD/src/lysozyme_in_water_singularityExecution.py \
	$PWD/config /gpfs/projects/bsc19/COMPSs_DATASETS/gromacs/$numMutation \
	$PWD/output
