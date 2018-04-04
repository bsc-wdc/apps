IT_HOME=/gpfs/apps/MN3/COMPSs/1.2/Runtime

$IT_HOME/scripts/queues/run.sh \
--app=/gpfs/projects/bsc19/COMPSs_APPS/kmeans/PyCOMPSs/0.4_nruns/src/0.4_kmeans.py \
--lang=python \
--classpath=/gpfs/projects/bsc19/COMPSs_APPS/kmeans/PyCOMPSs/0.4_nruns/src/:/home/bsc19/bsc19367/numpy-1.7.1/build/lib.linux-x86_64-2.7 \
--library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
--cline_args="1000 3 5 4" \
--num_nodes=2 \
--exec_time=5
