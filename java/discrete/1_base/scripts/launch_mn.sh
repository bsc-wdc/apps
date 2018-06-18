BASE_PATH=$HOME

enqueue_compss \
    --worker_working_dir=gpfs
    --lang=python
    --exec_time=5
    --num_nodes=1
    --classpath="${BASE_PATH}/apps/java/discrete/1_base/target/discrete.jar" \
    discrete.Discrete \
    true \
    3 \
    3 \
    3 \
    "${BASE_PATH}/apps/java/discrete/1_base/binary" \
    "${BASE_PATH}/apps/java/discrete/1_base/data" \
    "${BASE_PATH}/apps/java/discrete/1_base/data/1B6C" \
    "${BASE_PATH}/apps/java/discrete/1_base/tmp" \
    "${BASE_PATH}/apps/java/discrete/1_base/scores"
