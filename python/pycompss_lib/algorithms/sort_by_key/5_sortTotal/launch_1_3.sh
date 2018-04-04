#/bin/bash 

${IT_HOME}/scripts/user/enqueue_compss \
  --exec_time=10 \
  --num_nodes=$1 \
  --queue_system=lsf \
  --tasks_per_node=16 \
  --master_working_dir=. \
  --worker_working_dir=scratch \
  --lang=python \
  --classpath=/gpfs/projects/bsc19/COMPSs_APPS/sortByKey/python/0.5_sortTotal/src/ \
  --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
  --comm="es.bsc.compss.nio.master.NIOAdaptor" \
  --tracing=$2 \
  --graph=$2 \
  /gpfs/projects/bsc19/COMPSs_APPS/sortByKey/python/0.5_sortTotal/src/sort.py $3 $4
