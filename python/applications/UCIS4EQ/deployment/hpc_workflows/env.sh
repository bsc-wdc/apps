echo "CONDA:$CONDA_DEFAULT_ENV"
if [ ! -z "${CONDA_DEFAULT_ENV}" ]; then
	source deactivate salvus
fi
module purge
module load singularity
module load ANACONDA/5.0.1
module load intel impi
source activate salvus
#source activate /gpfs/projects/bsc44/earthquake/conda/salvus
module load TrunkJEA 
export PYTHONPATH=/gpfs/projects/bsc44/earthquake/pycompss:$PYTHONPATH
export SLIPGEN_IMAGE="/gpfs/projects/bsc44/earthquake/UCIS4EQ/slipgen/slipgen.simg"
module load fabric
export I_MPI_EXTRA_FILESYSTEM_LIST=gpfs
export I_MPI_EXTRA_FILESYSTEM=on
export SALVUS_BINARY=/gpfs/projects/bsc44/earthquake/UCIS4EQ/Salvus/bin/salvus
export SALVUS_PROCESSES=480
export SALVUS_PPN=48
