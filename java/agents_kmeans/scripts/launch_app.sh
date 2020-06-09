#!/bin/bash
if [ ! $# -eq 2 ]; then
  echo "./launch_app.sh MASTER_IP MASTER_PORT"
fi
MASTER_IP=$1
MASTER_REST_PORT=$2

echo "Requesting kmeans execution on ${MASTER_IP} through port ${MASTER_REST_PORT}"
compss_agent_call_operation --master_node=${MASTER_IP} --master_port=${MASTER_REST_PORT} --cei=kmeans.KMeansItf kmeans.KMeans 30000 40 200 20 2 1 2 true 0
