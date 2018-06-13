# If using --sklearn=True, better use python instead of enqueue_compss

#     --qos=debug \
#     --tracing=true \

enqueue_compss \
    --debug \
    --qos=debug \
    --tracing=true \
    --num_nodes=2 \
    --exec_time=120 \
    --cpus_per_node=48 \
    --worker_in_master_cpus=0 \
    --max_tasks_per_node=48 \
    --constraints=highmem \
    --scheduler="es.bsc.compss.scheduler.fifoDataScheduler.FIFODataScheduler" \
    --worker_working_dir=gpfs \
    ./main_dense.py \
        --regr=False \
        --path=/gpfs/projects/bsc19/COMPSs_DATASETS/random_forest \
        --name=1e6s1e2f3c \
        --sklearn=False \
        --n_estimators=96 \

# see sklearn docs for RandomForest arguments