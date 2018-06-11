#!/bin/bash
VERSION_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
APPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"

#COMPILE APP
cd ${VERSION_DIR}
mvn clean package

#LAUNCH
runcompss --classpath=${VERSION_DIR}/target/blastallone.jar blast.Blast true ${APP_DIR}/deps/binaries/blastall ${APPS_DIR}/datasets/Blast/databases/swissprot/swissprot ${APPS_DIR}/datasets/Blast/sequences/sargasso_test.fasta 4 /tmp/ /tmp/result.txt

#CLEAN DIRECTORY
mvn clean