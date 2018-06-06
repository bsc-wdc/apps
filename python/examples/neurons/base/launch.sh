
#/bin/bash 

${IT_HOME}/scripts/user/enqueue_compss \
  --exec_time=$1 \
  --num_nodes=$2 \
  --tasks_per_node=16 \
  --lang=python \
  --pythonpath=/gpfs/projects/bsc19/COMPSs_APPS/neurons/python/1.0_basic/ \
  --library_path=/gpfs/apps/MN3/INTEL/mkl/lib/intel64 \
  --tracing=$3 \
  /gpfs/projects/bsc19/COMPSs_APPS/neurons/python/1.0_basic/src/ns-data-proc_compss_objects.py $4 /gpfs/projects/bsc19/COMPSs_DATASETS/neurons/spikes.dat


