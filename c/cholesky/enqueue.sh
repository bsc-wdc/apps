enqueue_compss -t --lang=c --persistent_worker_c=true --qos=bsc_cs --cpus_per_node=48 --worker_in_master_cpus=0 --num_nodes=2 --worker_working_dir=/gpfs/scratch/bsc19/bsc19007 --exec_time=120 --output_profile=$(pwd)/cholesky.profile --appdir=$(pwd) $(pwd)/master/cholesky  $1 $2 


