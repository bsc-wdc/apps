
redis-server --daemonize yes

runcompss --lang=python \
--storage_impl=redis \
--storage_conf=$(pwd)/storage_conf.txt \
--pythonpath=$(pwd)/src \
--graph \
src/matmul.py --num_blocks 4 --elems_per_block 4 --check_result --seed 91


pkill redis
