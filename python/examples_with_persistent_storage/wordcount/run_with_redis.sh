# Set a standalone Redis backend
/usr/sbin/redis-server --daemonize yes

runcompss \
  -dg \
  --python_interpreter=python3 \
  --storage_impl=redis \
  --storage_conf=$(pwd)/redis_confs/storage_conf.cfg \
  src/wordcount_storage.py -d $(pwd)/dataset/

# End the storage standalone backend
pkill redis
