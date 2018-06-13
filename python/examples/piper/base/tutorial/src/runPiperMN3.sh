#!/bin/bash

PYAPPS=/gpfs/projects/bsc19/CRG
APPLICATION=${PYAPPS}/PIPER/piper_v2_pycompss.py

#export JAVA_HOME=/usr/lib/jvm/java-7-oracle
export IT_HOME=/gpfs/apps/MN3/COMPSs/Trunk/compss/compss-rt

export PATH=${PYAPPS}/PIPER/bin:${PYAPPS}/exonerate-2.2.0-x86_64/bin:${PYAPPS}/T-COFFE-Version_10.00.r1613/bin:/apps/BLAST/2.2.27+/bin:${PYAPPS}/PYTHON_2.7.6_WITH_BIOPYTHON/bin:${PATH}

set -x

${IT_HOME}/scripts/queues/run.sh \
--lang=python \
--classpath=${PYAPPS}/PYTHON_2.7.6_WITH_BIOPYTHON/lib/python2.7/site-packages:${PYAPPS}/PIPER/ \
--num_nodes=2 \
--exec_time=1 \
--app=${APPLICATION} \
--cline_args="" \
--debug=true \
--tracing=true

