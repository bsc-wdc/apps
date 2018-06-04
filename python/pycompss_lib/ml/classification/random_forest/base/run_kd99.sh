# If using --sklearn=True, better use python instead of enqueue_compss

#     --qos=debug \
#     --tracing=true \

enqueue_compss \
    --qos=debug \
    --tracing=true \
    --num_nodes=9 \
    --exec_time=120 \
    --cpus_per_node=48 \
    --worker_in_master_cpus=0 \
    --max_tasks_per_node=48 \
    --constraints=highmem \
    --scheduler="es.bsc.compss.scheduler.fifoDataScheduler.FIFODataScheduler" \
    --worker_working_dir=gpfs \
    ./main_kd99.py \
        --regr=False \
        --path=/gpfs/projects/bsc19/COMPSs_DATASETS/csvm/kdd99/ \
        --name=train.csv \
        --sklearn=False \
        --n_estimators=96 \

# see sklearn docs for RandomForest arguments