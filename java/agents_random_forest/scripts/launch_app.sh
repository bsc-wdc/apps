#!/bin/bash

if [ ! $# -eq 2 ]; then
  echo "./launch_app.sh MASTER_IP MASTER_PORT"
  exit 1
fi

MASTER_IP=$1
MASTER_REST_PORT=$2

echo "Requesting kmeans execution on ${MASTER_IP} through port ${MASTER_REST_PORT}"
compss_agent_call_operation \
    --master_node="${MASTER_IP}" \
    --master_port="${MASTER_REST_PORT}" \
    --cei=randomforest.RandomForestItf \
    randomforest.RandomForest 30000 40 200 20 2 1 2 true 0 10 10
