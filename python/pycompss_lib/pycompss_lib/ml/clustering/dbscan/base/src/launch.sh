enqueue_compss \
    --num_nodes=$1 \
    --tracing=$2 \
    --qos=$3 \
    --exec_time=$4 \
    --cpus_per_node=$5 \
    --worker_in_master_cpus=$6 \
    --scheduler=$7 \
    --worker_working_dir=$8 \
    ./DBSCAN.py \
        --is_mn=$9 \
        ${10} ${11} ${12}
# Expected elapsed time: around 3 minutes.
