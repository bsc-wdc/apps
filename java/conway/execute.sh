#!/bin/bash -e

runcompss  \
        -t \
        -g \
 --summary \
           \
    --classpath=./target/conway.jar \
    --project=./xml/project.xml     \
    --resources=./xml/resources.xml \
                                    \
    conway.blocks.Conway  2048 2048 4 1024

