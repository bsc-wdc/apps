#!/bin/bash

# File paths have to be adapted: writing permissions are required for working_worker_dir, base_log_dir and -o (output) files. The .pkl file is a pickled version of the dataset.
# Usage: ./launch_dr2.py <L> <minPts>

enqueue_compss --constraints=highmem --lang=python --scheduler=es.bsc.compss.scheduler.fifoDataScheduler.FIFODataScheduler --worker_in_master_cpus=0 --max_tasks_per_node=24 --worker_working_dir=/gpfs/scratch/bsc19/bsc19029/ --exec_time=2880 --num_nodes=3 --base_log_dir=/gpfs/scratch/bsc19/bsc19029/ /home/bsc19/bsc19029/gaia/clustering_dr2.py -L $1 -m $2 -o /home/bsc19/bsc19029/gaia/out_dr2 /gpfs/projects/bsc19/COMPSs_DATASETS/dislib/gaia/df_dr2.pkl 
