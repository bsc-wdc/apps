#!/bin/bash

redis-server --daemonize yes

runcompss \
--storage_impl=redis \
--pythonpath=$(pwd)/src \
--storage_conf=$(pwd)/storage_conf.txt \
$(pwd)/src/kmeans.py --mode normal --plot_result --seed 7 --mode uniform

pkill redis
