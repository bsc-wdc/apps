#!/bin/bash
numNodes=$1
numMutation=$2
export MDCU=48
export GMX_BIN=/gpfs/projects/bsc19/TUTORIAL/gromacs5.1.2/bin
export PATH=$GMX_BIN:$PATH
export COMPSS_PYTHON_VERSION=3
module load TrunkCAA
module load singularity/3.5.2
module load grace
#module load intel/2018.4 impi/2018.4 mkl/2018.4 gromacs/2019.1
JAVA_TOOL_OPTIONS=-Xss1280k
enqueue_compss --max_tasks_per_node=1 --pythonpath=/home/bsc19/bsc19286/gromacs/src -t \
	--python_interpreter=python3 \
	--qos=debug --num_nodes=$numNodes --worker_in_master_cpus=48 --worker_working_dir=gpfs \
	/home/bsc19/bsc19286/gromacs/src/lysozyme_in_water_onlyBinaryExecution.py \
	/home/bsc19/bsc19286/gromacs/config /home/bsc19/bsc19286/gromacs/dataset/$numMutation \
	/home/bsc19/bsc19286/gromacs/output
