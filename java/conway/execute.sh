#!/bin/bash -e

  runcompss \
    -d \
    \
    -g \
    -t \
    --summary \
    \
    --classpath=./target/conway.jar \
    --project=./xml/project.xml \
    --resources=./xml/resources.xml \
    \
    conway.blocks.Conway 64 64 4 32

