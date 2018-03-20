#!/bin/bash

if [ $# -ne 1 ]; then
	echo "Bad arguments. Usage: $0 <file_or_dir_to_copy>"
	exit -1
fi

if [ ! -e $1 ]; then
	echo "Error. Path $1 does not exist."
	exit -1
fi
	

docker cp $1 dockers_ds1java_1:tmp
docker cp $1 dockers_ds2java_1:tmp
