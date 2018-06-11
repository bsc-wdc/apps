#!/bin/bash
VERSION_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
APPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"

#COMPILE APP
cd ${VERSION_DIR}
mvn clean package

#LAUNCH
export HMMER_BINARY=${APP_DIR}/deps/binaries/hmmpfam
runcompss --classpath=${VERSION_DIR}/target/hmmerobj.jar hmmerobj.HMMPfam  ${APPS_DIR}/datasets/Hmmer/smart.HMMs.bin ${APPS_DIR}/datasets/Hmmer/256seq /tmp/hmmer.result 4 4

#CLEAN DIRECTORY
mvn clean
