export COMPSS_PYTHON_VERSION=3.12.1
module load COMPSs/3.3.3 dislib/master

enqueue_compss \
    --num_nodes=1 \
    --exec_time=60 \
    --tracing=true \
    -g \
    --project_name=bsc19 \
    --log_dir=$(pwd) \
    --master_working_dir=$(pwd) \
    --worker_working_dir=$(pwd) \
    --qos=gp_debug \
    --log_level=debug \
    --pythonpath=$(pwd) \
    --python_interpreter=$(which python3) \
    --lang=python Gauss_ds.py
