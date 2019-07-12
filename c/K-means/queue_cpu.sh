#!/bin/bash

enqueue_compss --qos=bsc_cs --persistent_worker_c=true --lang=c                             \
	--cpus_per_task --exec_time=120 --num_nodes=1 --cpus_per_node=160     \
	--worker_in_master_cpus=160 --worker_working_dir=/gpfs/scratch/bsc19/bsc19007/          \
	master/kmeans -i /gpfs/projects/bsc19/COMPSs_APPS/c/K-means/generator/N200000_K50_d50_0.txt -n 200000 -f 8  -l 2 -k 50 -d 50 -o 
    
