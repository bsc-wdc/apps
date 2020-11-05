#!/bin/bash

enqueue_compss -t --jvm_workers_opts="-Dcompss.worker.removeWD=false" \
	--output_profile=out.prof --persistent_worker_c=true --queue=debug --lang=c \
	--exec_time=30 --num_nodes=2 --cpus_per_node=16 --gpus_per_node=4 \
	--worker_in_master_cpus=0 --worker_working_dir=$PWD \
	master/kmeans -i /gpfs/projects/bsc19/COMPSs_APPS/c/K-means/generator/N200000_K50_d50_0.txt -n 200000 -f 30 -l 4 -k 50 -d 50 -c 24 -o 
