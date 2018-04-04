#/bin/bash 

${IT_HOME}/scripts/user/enqueue_compss \
  --exec_time=300 \
  --num_nodes=3 \
  --queue_system=lsf \
  --tasks_per_node=16 \
  --master_working_dir=. \
  --worker_working_dir=scratch \
  --lang=python \
  --classpath=/gpfs/projects/bsc19/COMPSs_APPS/kmeans/python/1.0_spark/src/ \
  --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
  --comm="es.bsc.compss.nio.master.NIOAdaptor" \
  --tracing=true \
  --graph=true \
  /gpfs/projects/bsc19/COMPSs_APPS/kmeans/python/1.0_spark/src/kmeans_frag_file2.py $1 $2 $3 $4 $5 $6
