#/bin/bash 

${IT_HOME}/scripts/user/enqueue_compss \
  --exec_time=$1 \
  --num_nodes=$2 \
  --queue_system=lsf \
  --tasks_per_node=$3 \
  --master_working_dir=. \
  --worker_working_dir=$4 \
  --lang=python \
  --classpath=/gpfs/projects/bsc19/COMPSs_APPS/wordcount/python/0.1_base/lectura_worker/src/ \
  --comm=$5 \
  --tracing=$6 \
  /gpfs/projects/bsc19/COMPSs_APPS/wordcount/python/0.1_base/lectura_worker/src/wordcount.py $7 wordcount.out $8

#  --comm="es.bsc.compss.gat.master.GATAdaptor"
#  --comm="es.bsc.compss.nio.master.NIOAdaptor"
