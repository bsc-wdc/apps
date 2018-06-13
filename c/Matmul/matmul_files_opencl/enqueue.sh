#enqueue_compss -d --lang=c --gpus_per_node=1 --project=/home/bsc19/bsc19430/increment/xml/project.xml --resources=/home/bsc19/bsc19430/increment/xml/default_resources.xml --appdir=/home/bsc19/bsc19430/increment/master/increment /home/bsc19/bsc19430/increment/master/increment 4 1 2 3

#enqueue_compss -d --worker_working_dir=gpfs --lang=c --appdir=/home/bsc19/bsc19430/matmul_objects_ompss/ /home/bsc19/bsc19430/matmul_objects_ompss/master/Matmul 8 64 12.34

enqueue_compss -d -m --lang=c --num_nodes=2 --gpus_per_node=2 --tasks_per_node=12 --exec_time=90 --appdir=/home/bsc19/bsc19430/matmul_files_opencl /home/bsc19/bsc19430/matmul_files_opencl/master/Matmul 1 64 12.34

