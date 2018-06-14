#!/bin/bash
VERSION_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
APPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"

#COMPILE APP
cd ${VERSION_DIR}/application
mvn clean package

#LAUNCH
MASTER_DIR=`mktemp -d `
cd $MASTER_DIR
export HMMER_BINARY=${APP_DIR}/deps/binaries/hmmpfam

runcompss \
	--classpath=${VERSION_DIR}/application/target/hmmerws.jar \
	--project=${VERSION_DIR}/conf/project.xml \
	--resources=${VERSION_DIR}/conf/resources.xml \
	hmmerws.HMMPfam  ${APPS_DIR}/datasets/Hmmer/smart.HMMs.bin ${APPS_DIR}/datasets/Hmmer/256seq /tmp/hmmer.result 4 4

#CLEAN DIRECTORY
rm -rf ${MASTER_DIR}
cd ${VERSION_DIR}/application
mvn clean
