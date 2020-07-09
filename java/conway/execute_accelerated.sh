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
    conway.accelerated.Conway 512 512 400 256 79