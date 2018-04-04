IT_HOME=/gpfs/apps/MN3/COMPSs/1.2/Runtime

$IT_HOME/scripts/queues/run.sh \
--app=/gpfs/projects/bsc19/COMPSs_APPS/wordcount/PyCOMPSs/0.2_merge/lectura_worker/src/wordcount.py \
--lang=python \
--classpath=/gpfs/projects/bsc19/COMPSs_APPS/wordcount/PyCOMPSs/0.2_merge/lectura_worker/src/ \
--library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
--cline_args="/gpfs/projects/bsc19/COMPSs_APPS/wordcount/data/all.txt wordcount.out 10000000" \
--exec_time=15 \
--num_nodes=2
