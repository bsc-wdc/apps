#!/bin/bash

echo ""
if [ "x$1" == "x" ]; then
    echo "Usage: $0 <pattern> [e.g. \"mass -A3\"]"
else
    echo "Consumer results"
    echo "================"
    echo ""
    for i in `ls $HOME/IT/severo.consumer.Consumer/jobs/job*.out`; do
        cat $i | grep $1 
    done
fi
echo ""
