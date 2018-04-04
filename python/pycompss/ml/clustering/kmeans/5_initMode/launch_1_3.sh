#/bin/bash 

${IT_HOME}/scripts/user/enqueue_compss \
  --debug=debug \
  --exec_time=$1 \
  --num_nodes=$2 \
  --queue_system=lsf \
  --tasks_per_node=$3 \
  --master_working_dir=. \
  --worker_working_dir=$4 \
  --lang=python \
  --classpath=/gpfs/projects/bsc19/COMPSs_APPS/kmeans/python/0.5_initMode/src/ \
  --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
  --comm=$5 \
  --tracing=$6 \
  /gpfs/projects/bsc19/COMPSs_APPS/kmeans/python/0.5_initMode/src/kmeans_frag_v3.py $7 $8 $9 ${10} ${11}
