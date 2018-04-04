IT_HOME=/gpfs/apps/MN3/COMPSs/1.2/Runtime

$IT_HOME/scripts/queues/run.sh \
--app=/gpfs/projects/bsc19/COMPSs_APPS/wordcount/python/0.3_files/lectura_master/src/wordcount.py \
--lang=python \
--classpath=/gpfs/projects/bsc19/COMPSs_APPS/wordcount/python/0.3_files/lectura_master/src/ \
--library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
--cline_args="$4 wordcount.out" \
--exec_time=$1 \
--num_nodes=$2 \
--tracing=$3 \
--graph=true
