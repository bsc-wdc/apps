#!/bin/bash -e

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  APP_DIR=${SCRIPT_DIR}
  SRC_DIR=${APP_DIR}/base/src/

  A_MINI_HOME="/gpfs/projects/bsc19/COMPSs_APPS/ngsa_mini/ngsa-mini-py"
  INPUT_PATH="/gpfs/projects/bsc19/COMPSs_DATASETS/ngsa_mini"

  # Enqueue options
  workingDir=gpfs
  numNodes=2
  execTime=60

  # Enqueue job
  enqueue_compss \
    --exec_time=$execTime \
    --num_nodes=$numNodes \
    \
    --log_level=debug \
    --summary \
    --tracing=false \
    --graph=false \
    \
    --worker_working_dir=$workingDir \
    --appdir="${APP_DIR}" \
    --pythonpath="${SRC_DIR}" \
    \
    "${SRC_DIR}"/workflow.py $INPUT_PATH/bwa_db/reference.fa $INPUT_PATH/seq_contig.md $INPUT_PATH/reference.fa $INPUT_PATH/reference.fa.fai $INPUT_PATH/work/wfinput_08_8000000 8
