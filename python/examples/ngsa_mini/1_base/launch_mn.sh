#!/bin/bash -e

# Define script variables
NGSA_MINI_HOME="/gpfs/projects/bsc19/COMPSs_APPS/ngsa_mini/ngsa-mini-py"
INPUT_PATH="/gpfs/projects/bsc19/COMPSs_DATASETS/ngsa_mini"
execFile=${NGSA_MINI_HOME}/1_base/src/workflow.py
appClasspath=${NGSA_MINI_HOME}/1_base/src
appPythonpath=${NGSA_MINI_HOME}/1_base/src
WORK_DIR=${NGSA_MINI_HOME}/1_base/work_compss

# Just in case, remove previous output data
rm -rf time.txt
rm -rf work/*

# Retrieve arguments
if [ $# -lt "1" ]; then
  n=8
else
  n=$1
fi

if [ $n -eq "16" ]; then
  input_folder=${INPUT_PATH}/work/wfinput_16_16000000
  num_nodes=2
  max_time=40
elif [ $n -eq "32" ]; then
  input_folder=${INPUT_PATH}/work/wfinput_32_32000000
  num_nodes=4
  max_time=80
elif [ $n -eq "64" ]; then
  input_folder=${INPUT_PATH}/work/wfinput_64_64000000
  num_nodes=8
  max_time=160
else
  input_folder=${INPUT_PATH}/work/wfinput_08_8000000
  num_nodes=1
  max_time=20
fi

echo "Calling NGSA-mini with "${n}" tasks and "${num_nodes}" nodes"

# Enqueue the application
enqueue_compss \
  --max_tasks_per_node=8 \
  --job_dependency=None \
  --num_nodes=$num_nodes \
  --worker_in_master_memory=80000 \
  --worker_in_master_cpus=48 \
  --exec_time=$max_time \
  --master_working_dir=. \
  --worker_working_dir=/gpfs/projects/bsc19/COMPSs_APPS/ngsa_mini/ngsa-mini-py/ \
  --tracing=false \
  --classpath=$appClasspath \
  --pythonpath=$appPythonpath \
  -gt \
  --lang=python \
  $execFile \
  ${INPUT_PATH}/bwa_db/reference.fa \
  ${INPUT_PATH}/seq_contig.md \
  ${INPUT_PATH}/reference.fa \
  ${INPUT_PATH}/reference.fa.fai \
  $input_folder \
  $WORK_DIR \
  $n

#### HOW TO EXECUTE
# 
# Syntax:
# ./launch.sh [NUM_JOBS <8,16,32,64>]
# 
# Example:
# ./launch.sh 8

