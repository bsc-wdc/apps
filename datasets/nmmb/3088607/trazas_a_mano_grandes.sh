#BSUB -W 90
#BSUB -J merge
#BSUB -cwd /gpfs/projects/bsc19/bsc19234/nmmb/OUTPUT/CURRENT_RUN
#BSUB -eo err
#BSUB -oo out
#BSUB -n 10

module load openmpi
module load extrae

mpirun -np 10 mpimpi2prv -f /gpfs/projects/bsc19/bsc19234/nmmb/OUTPUT/CURRENT_RUN/TRACE.mpits -o /gpfs/projects/bsc19/bsc19234/nmmb/OUTPUT/CURRENT_RUN/mpitask.prv


