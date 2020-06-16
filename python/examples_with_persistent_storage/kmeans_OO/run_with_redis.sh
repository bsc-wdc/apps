# Set a standalone Redis backend
/usr/sbin/redis-server --daemonize yes

runcompss \
  -d \
  --python_interpreter=python3 \
  --pythonpath=$(pwd)/src \
  --storage_impl=redis \
  --storage_conf=$(pwd)/redis_confs/storage_conf.cfg \
  src/kmeans.py -n 1024 -f 8 -d 2 -c 4 --use_storage

# End the storage standalone backend
pkill redis
