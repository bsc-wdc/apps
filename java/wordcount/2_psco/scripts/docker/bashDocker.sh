#!/bin/bash

if [ $# -ne 1 ]; then
	echo "Bad arguments. Usage: $0 <image_name>"
	exit -1
fi

docker exec -i -t $1 /bin/bash
