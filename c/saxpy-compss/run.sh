#!/bin/bash
PROGRAM=saxpy-p

#export NX_ARGS=" --force-tie-master "
export NX_ARGS=" --verbose --verbose-devops"
export NX_SMP_WORKERS="1"
export NX_SMP_PRIVATE_MEMORY=yes
#export NX_INSTRUMENTATION=tdg
#export NX_DEPS=cregions
#export NX_OPENCL_CACHE_POLICY=nocache
#export NX_OPENCL_CACHE_POLICY = nocache
#export NX_OPENCL_MAX_DEVICES=2 #max number of opencl devices (GPUs in this case) to use

#valgrind --tool=callgrind ./$PROGRAM
./$PROGRAM

