

# Set a single standalone Redis instance
redis-server --daemonize yes


runcompss --lang=python \
--pythonpath=$(pwd)/src:$(pwd)/COMPSs-Redis-bundle/python \
--classpath=$(pwd)/COMPSs-Redis-bundle/compss-redisPSCO.jar \
--storage_conf=$(pwd)/storage_conf.txt \
--debug \
--graph \
src/matmul.py 4 8 4 16 true

pkill redis
