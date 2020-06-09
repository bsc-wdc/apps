#!/bin/bash -e

user=${1:-bsc19533}
mn_login=${2:-mn1.bsc.es}
target_folder=${3:-/gpfs/projects/bsc19/agents_apps/kmeans}

# shellcheck disable=SC2029
ssh "${user}@${mn_login}" rm -rf "${target_folder}/*"

scp -r scripts "${user}@${mn_login}:${target_folder}"
scp -r src/ "${user}@${mn_login}:${target_folder}"
scp -r pom.xml "${user}@${mn_login}:${target_folder}"
scp -r README.md "${user}@${mn_login}:${target_folder}"

# shellcheck disable=SC2029
ssh "${user}@${mn_login}" chmod 777 -R "${target_folder}"

# Done
echo "DONE"

