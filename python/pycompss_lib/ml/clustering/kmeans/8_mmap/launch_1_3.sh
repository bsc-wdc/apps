#/bin/bash 

${IT_HOME}/scripts/user/enqueue_compss \
  --exec_time=100 \
  --num_nodes=$1 \
  --queue_system=lsf \
  --tasks_per_node=16 \
  --master_working_dir=. \
  --worker_working_dir=scratch \
  --lang=python \
  --classpath=/gpfs/projects/bsc19/COMPSs_APPS/kmeans/python/1.1_merge/src/ \
  --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
  --comm="es.bsc.compss.nio.master.NIOAdaptor" \
  --tracing=$2 \
  --graph=$2 \
  /gpfs/projects/bsc19/COMPSs_APPS/kmeans/python/1.2_mmap/src/kmeans_spark.py $3 $4 $5 $6 $7 $8
