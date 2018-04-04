#/bin/bash 

${IT_HOME}/scripts/user/enqueue_compss \
  --exec_time=15 \
  --num_nodes=3 \
  --queue_system=lsf \
  --tasks_per_node=16 \
  --master_working_dir=. \
  --worker_working_dir=scratch \
  --lang=python \
  --classpath=/gpfs/projects/bsc19/COMPSs_APPS/kmeans/PyCOMPSs/0.2_2steps/ \
  --comm="es.bsc.compss.nio.master.NIOAdaptor" \
  /gpfs/projects/bsc19/COMPSs_APPS/kmeans/PyCOMPSs/0.2_2steps/src/0.2_kmeans.py 1000 3 5 4

#  --comm="es.bsc.compss.gat.master.GATAdaptor"
#  --comm="es.bsc.compss.nio.master.NIOAdaptor"
