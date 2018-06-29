#!/bin/bash

redis-server --daemonize yes

runcompss \
--storage_impl=redis \
--pythonpath=$(pwd)/src \
--storage_conf=$(pwd)/storage_conf.txt \
-t \
-g \
$(pwd)/src/kmeans.py --numpoints 10000 --mode normal --plot_result --seed 7

pkill redis
