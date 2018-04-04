#/bin/bash 

${IT_HOME}/scripts/user/enqueue_compss \
  --exec_time=5\
  --num_nodes=$1 \
  --queue_system=lsf \
  --tasks_per_node=16 \
  --master_working_dir=. \
  --worker_working_dir=scratch \
  --lang=python \
  --classpath=/gpfs/projects/bsc19/COMPSs_APPS/sortByKey/python/1.0_sortByKey/src/ \
  --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
  --comm="es.bsc.compss.nio.master.NIOAdaptor" \
  --tracing=$2 \
  --graph=$3 \
  /gpfs/projects/bsc19/COMPSs_APPS/sortByKey/python/1.0_sortByKey/src/sort.py 10 5 3 5 2 5 12345 false undefined
  #/gpfs/projects/bsc19/COMPSs_APPS/sortByKey/python/1.0_sortByKey/src/sort.py $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}

#python sort.py 10 5 3 5 2 5 12345 false


#/gpfs/projects/bsc19/COMPSs_APPS/sortByKey/python/1.0_sortByKey/src/sort.py $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}
