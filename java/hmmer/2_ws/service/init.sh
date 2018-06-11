#!/bin/bash
SERVICE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKING_DIR="/tmp/hmmerService"

rm -rf ${WORKING_DIR}
mkdir -p ${WORKING_DIR}
cd ${WORKING_DIR}


echo "Compiling Service classes";
javac -d ${WORKING_DIR} ${SERVICE_DIR}/worker/hmmerobj/*.java

echo "Generating WSDL"
wsgen -cp ${WORKING_DIR} worker.hmmerobj.HmmerObjectsImpl -wsdl

echo "Compiling Service Listener";
javac -cp ${WORKING_DIR} -d ${WORKING_DIR} ${SERVICE_DIR}/worker/hmmerobj/publish/HmmerObjectsImplServicePublisher.java

echo "Running Service";
java -cp ${WORKING_DIR} worker.hmmerobj.publish.HmmerObjectsImplServicePublisher

echo "Cleaning";
rm -rf ${WORKING_DIR}

