#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/kmeans.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  WORKER_WORKING_DIR=scratch
  WORK_DIR=${scriptDir}/results/

  # Storage-related paths
  # Change these paths if you want to use other storage implementations
  storage_home=${COMPSS_HOME}/Tools/storage/redis
  storage_classpath=${storage_home}/compss-redisPSCO.jar

  # Retrieve arguments
  jobDependency=$1
  numNodes=$2
  executionTime=$3
  tasksPerNode=$4
  tracing=$5

  # Leave application args on $@
  shift 5

  export ComputingUnits=12

  # Enqueue the application
  enqueue_compss \
    --job_dependency=$jobDependency \
    --num_nodes=$numNodes \
    --cpus_per_node=$tasksPerNode \
    --exec_time=$executionTime \
    --master_working_dir=$WORK_DIR \
    --worker_working_dir=$WORKER_WORKING_DIR \
    --classpath=$appClasspath:$storage_classpath \
    --pythonpath=$appPythonpath:$storage_home/python \
    --storage_props=$storage_home/scripts/sample_props.cfg \
    --storage_home=$storage_home/ \
    --lang=python \
    --tracing=$tracing \
    $execFile $@

  # $@ should contain all the app arguments
  # The available app arguments are:
  # -s / --seed Pseudo random seed
  # -n / --num_points Number of points
  # -d / --dimensions Dimensions of the points
  # -c / --centres Number of centres
  # -f / --fragments Number of fragments
  # -m / --mode Uniform or normal
  # -i / --iterations Number of MAXIMUM iterations
  # -e / --epsilon Epsilon tolerance
  # -l / --lnorm Norm of vectors (l1 or l2)
  # --plot_result Plot clustering. Only works if dimensions = 2
  # --use_storage

  # ./launch.sh None 3 5 16 false 160 3 4 4
