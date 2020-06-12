#!/bin/bash

# deploy_app.sh --> deploys 2 agent on port 46000
# deploy_app.sh N --> deploys 2 agent on port P
# deploy_app.sh N P--> deploys N agents from port P to P+N-1

# GLOBAL VARS
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Parse arguments
if [ $# -eq 0 ]; then
  port=46000
  numWorkers=2
else
  numWorkers=$1
  if [ $# -eq 1 ]; then
    port=46000
  else
    port=$2
  fi    
fi

# Main
echo "    Closing previous Yakuake sessions"
ACTIVE_SESSIONS=$(qdbus org.kde.yakuake /yakuake/sessions sessionIdList)
OLD_IFS=${IFS}
IFS=',' read -ra ADDR <<< "${ACTIVE_SESSIONS}"
for SESSION in "${ADDR[@]}"; do
  SESSION_TITLE=$(qdbus org.kde.yakuake /yakuake/tabs tabTitle "${SESSION}")
  if [[ "${SESSION_TITLE}" == Agent* ]]; then
    qdbus org.kde.yakuake /yakuake/sessions removeSession "${SESSION}" > /dev/null
  fi
done
IFS=${OLD_IFS}


echo "    Start agents..."
APP_DIR="${SCRIPT_DIR}/../target/simple_files.jar"
for i in $(seq 1 "${numWorkers}"); do
  WORKER_IP="127.0.0.${i}"
  REST_PORT=$(( port + "${i}" *100 + 1))
  COMM_PORT=$(( port + "${i}" *100 + 2))
  echo "        Starting agent ${WORKER_IP} on ports ${REST_PORT}(REST) and ${COMM_PORT}(comm)"
  WORKER_SESSION=$(qdbus org.kde.yakuake /yakuake/sessions addSession)
  qdbus org.kde.yakuake /yakuake/tabs setTabTitle "${WORKER_SESSION}" "Agent${i}" > /dev/null
  qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommandInTerminal "${WORKER_SESSION}" "\
    compss_agent_start --hostname=${WORKER_IP} -d --classpath=${APP_DIR} --rest_port=${REST_PORT} --comm_port=${COMM_PORT}
  " >/dev/null 2>/dev/null
done

echo "    Waiting for agents to be ready..."
CONFIRMED_AGENTS=0
while [ ${CONFIRMED_AGENTS} -lt "${numWorkers}" ]; do
  CONFIRMED_AGENTS=0
  for i in $(seq 1 "${numWorkers}"); do
    WORKER_IP="127.0.0.${i}"
    REST_PORT=$(( port + "${i}" * 100 + 1))
    curl -XGET "http://${WORKER_IP}:${REST_PORT}/COMPSs/test" 2>/dev/null
    ev=$?
    if [ "$ev" -eq 0 ]; then
      CONFIRMED_AGENTS=$(( CONFIRMED_AGENTS + 1))
    fi
  done
done
sleep 1
echo "    Agents are up and running!"

i=1
MASTER_IP="127.0.0.${i}"
MASTER_REST_PORT=$(( port + "${i}" *100 + 1))
if [ "${numWorkers}" -gt 1 ]; then
  echo "    Adding resources to ${MASTER_IP}"
  for i in $(seq 2 "${numWorkers}"); do
    WORKER_IP="127.0.0.${i}"
    COMM_PORT=$(( port + "${i}" * 100 + 2))
    echo "        Adding ${WORKER_IP} on port ${COMM_PORT}"
   
    compss_agent_add_resources "--agent_node=${MASTER_IP}" "--agent_port=${MASTER_REST_PORT}" "${WORKER_IP}" "Port=${COMM_PORT}" >/dev/null
  done
fi

curl -XGET http://${MASTER_IP}:${MASTER_REST_PORT}/COMPSs/printResources 2>/dev/null
