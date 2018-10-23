BASE_PATH=$HOME

enqueue_compss \
    --worker_working_dir=scratch
    --lang=python
    --exec_time=5
    --num_nodes=1
    --pythonpath="${BASE_PATH}/apps/python" \
    csvm-driver.py \
        -i 5 \
        --convergence \
        /gpfs/projects/bsc19/COMPSs_DATASETS/csvm/agaricus/train.csv \
        -t /gpfs/projects/bsc19/COMPSs_DATASETS/csvm/agaricus/test.csv \