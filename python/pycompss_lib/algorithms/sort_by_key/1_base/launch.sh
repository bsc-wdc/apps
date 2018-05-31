#!/bin/bash

function help {
    echo "Usage: $0"
    echo "-v/--version             1.2 | 1.3"
    echo "-n/--nodes               <int>"
    echo "-t/--time                <int>"
    echo "[-c/--comm               NIO | GAT]      default: NIO"
    echo "[-tn/--task-x-node       <int>]          default: 8"
    echo "[-w/--worker-working-dir scratch | gpfs] default: scratch"
    echo "[-tr/--tracing           true | false]   default: false"
    echo "[-a/--args"
    exit
}

while [ ! $# -eq 0 ]
do
    case "$1" in
        --help | -h)
            help
            ;;
        --task-x-node | -tn)
            shift
            TXN=$1
            echo $TXN
            ;;
        --time | -t)
            shift
            TIME=$1
            echo $TIME
            ;;
        --version | -v)
            shift
            VERSION=$1
            ;;
        --worker-working-dir | -w)
            shift
            WWD=$1
            ;;
        --comm | -c)
            shift
            COMM=$1
            ;;
        --nodes | -n)
            shift
            NODES=$1
            ;;
        --tracing | -tr)
            shift
            TRACE=$1
            ;;
        --args | -a)
            shift
            ARGS=$*
            ;;
    esac
    shift
done

if [ "$COMM" = ""  ]; then
    COMM="NIO"
    echo set comm $COMM
fi
if [ "$WWD" = "" ]; then
    WWD="scratch"
    echo set worker working dir to $WWD
fi
if [ "$TXN" = "" ]; then
    TXN="8"
    echo set task per node to $TXN
fi

if [ "$TRACE" = "" ]; then
    TRACE="false"
    echo set trace to $TRACE
fi

if [ "$COMM" = "NIO" ]
then
   COMM="es.bsc.compss.nio.master.NIOAdaptor"
else
   COMM="es.bsc.compss.gat.master.GATAdaptor"
fi

echo $ARGS

if [ "$VERSION" = "1.2" ]; then
    ./launch_1_2.sh $TIME $NODES $TRACE $ARGS
else
    ./launch_1_3.sh $TIME $NODES $TXN $WWD $COMM $TRACE $ARGS
fi
