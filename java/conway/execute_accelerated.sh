#!/bin/bash -e

runcompss  \
        -g \
        -t \
 --summary \
           \
    --classpath=./target/conway.jar \
    --project=./xml/project.xml     \
    --resources=./xml/resources.xml \
                                    \
    conway.accelerated.Conway  2048 2048 400 512 79

