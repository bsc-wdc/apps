#!/bin/bash

PYAPPS=${HOME}/CRG/
APPLICATION=${PYAPPS}/PIPER/src/piper_v2_pycompss.py

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
export IT_HOME=/opt/COMPSs/Runtime
export GAT_LOCATION=/opt/COMPSs/JAVA_GAT

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${IT_HOME}/bindings/bindings-common/lib

export PATH=${PYAPPS}/PIPER/bin:${PYAPPS}/exonerate-2.2.0-x86_64/bin:${PYAPPS}/T-COFFEE-Version_10.00.r1613/bin:${PYAPPS}/ncbi-blast-2.2.27+-src/c++/GCC480-Debug64/bin:${PYAPPS}/PYTHON_2.7.6_WITH_BIOPYTHON/bin:${PATH}

set -x

${IT_HOME}/scripts/user/runcompssext \
--lang=python \
--classpath=${PATH}:${PYAPPS}/PYTHON_2.7.6_WITH_BIOPYTHON/lib/python2.7:${PYAPPS}/PYTHON_2.7.6_WITH_BIOPYTHON/lib/python2.7/site-packages:${PYAPPS}/PIPER/src \
--app=${APPLICATION} \
--cline_args="" \
--project=${PYAPPS}/PIPER/src/project.xml \
--resources=${PYAPPS}/PIPER/src/resources.xml \
--tracing=true

