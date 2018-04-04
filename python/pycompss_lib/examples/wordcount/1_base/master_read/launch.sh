#/bin/bash 

enqueue_compss \
  --exec_time=$1 \
  --num_nodes=$2 \
  --tasks_per_node=16 \
  --master_working_dir=. \
  --worker_working_dir=scratch \
  --lang=python \
  --pythonpath=/gpfs/projects/bsc19/COMPSs_APPS/wordcount/python/0.1_base/lectura_master/src/ \
  --tracing=$3 \
  /gpfs/projects/bsc19/COMPSs_APPS/wordcount/python/0.1_base/lectura_master/src/wordcount.py /gpfs/projects/bsc19/COMPSs_DATASETS/wordcount/all.txt wordcount.out 10000000
