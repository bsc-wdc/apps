#/bin/bash 

${IT_HOME}/scripts/user/enqueue_compss \
  --exec_time=15 \
  --num_nodes=3 \
  --queue_system=lsf \
  --tasks_per_node=16 \
  --master_working_dir=. \
  --worker_working_dir=scratch \
  --lang=python \
  --classpath=/gpfs/projects/bsc19/COMPSs_APPS/wordcount/python/0.2_merge/lectura_worker/src/ \
  --comm=es.bsc.compss.nio.master.NIOAdaptor \
  /gpfs/projects/bsc19/COMPSs_APPS/wordcount/python/0.2_merge/lectura_worker/src/wordcount.py /gpfs/projects/bsc19/COMPSs_APPS/wordcount/data/all.txt wordcount.out  10000000

#  --comm="es.bsc.compss.gat.master.GATAdaptor"
#  --comm="es.bsc.compss.nio.master.NIOAdaptor"
