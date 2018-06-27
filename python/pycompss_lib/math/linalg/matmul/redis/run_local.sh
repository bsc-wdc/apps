
redis-server --daemonize yes

runcompss --lang=python \
--storage_impl=redis \
--storage_conf=$(pwd)/storage_conf.txt \
--pythonpath=$(pwd)/src \
--debug \
--graph \
src/matmul.py 4 2 4 16 true


pkill redis
