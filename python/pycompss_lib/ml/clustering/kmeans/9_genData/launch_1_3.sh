#/bin/bash 

${IT_HOME}/scripts/user/enqueue_compss \
  --exec_time=40 \
  --num_nodes=3 \
  --queue_system=lsf \
  --tasks_per_node=16 \
  --master_working_dir=. \
  --worker_working_dir=scratch \
  --lang=python \
  --classpath=/gpfs/projects/bsc19/COMPSs_APPS/kmeans/python/1.3_genData/src/ \
  --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
  --comm="es.bsc.compss.nio.master.NIOAdaptor" \
  --tracing=true \
  --graph=true \
  /gpfs/projects/bsc19/COMPSs_APPS/kmeans/python/1.3_genData/src/kmeans.py $1 $2 $3 $4

#  numV = numero de punts
#  dim = dimensions del punt
#  k = numero de centres
#  numFrag = numero de fragments 
