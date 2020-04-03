# Set a standalone Redis backend
/usr/sbin/redis-server --daemonize yes

runcompss \
  -d \
  --python_interpreter=python3 \
  --storage_impl=redis \
  --storage_conf=$(pwd)/redis_confs/storage_conf.cfg \
  src/hello_world.py

# End the storage standalone backend
pkill redis
