# If using --sklearn=True, better use python instead of enqueue_compss

#     --qos=debug \
#     --tracing=true \

enqueue_compss \
    --num_nodes=16 \
    --exec_time=30 \
    --cpus_per_node=48 \
    --worker_in_master_cpus=24 \
    --max_tasks_per_node=48 \
    --scheduler="es.bsc.compss.scheduler.fifoDataScheduler.FIFODataScheduler" \
    --worker_working_dir=scratch \
    ./main_RF.py \
        --regr=False \
        --path=/gpfs/projects/bsc19/COMPSs_DATASETS/random_forest/ \
        --name=0 \
        --sklearn=False \

# see sklearn docs for RandomForest arguments