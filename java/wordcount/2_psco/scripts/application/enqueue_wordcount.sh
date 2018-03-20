#!/bin/bash

  #-----------------------------------------------------
  # SET APPLICATION VARIABLES
  #-----------------------------------------------------
  STORAGE_HOME=/apps/COMPSs/DATACLAY/

  APPNAME="Wordcount"
  APP_HOME=$HOME/$APPNAME/
  DATASET="${APPNAME}DS"
  APPUSER="Consumer"

  #-----------------------------------------------------
  # COPY VARIABLES TO PROPS FILE
  #-----------------------------------------------------
  propsFile=$(mktemp -p ${APP_HOME}/scripts/application/)
  cat > $propsFile << EOT
APPUSER=$APPUSER
DATASET=$DATASET
EOT

  #-----------------------------------------------------
  # SET EXECUTION VARIABLES
  #-----------------------------------------------------
  jobsDir=$HOME/.COMPSs/\${LSB_JOBID}/storage/
  cfgsDir=${jobsDir}/cfgfiles/
  stubsDir=${jobsDir}/stubs/
  storageConf=${cfgsDir}/storage.properties
  clientConf=${cfgsDir}/client.properties
  cp="${stubsDir}:${APP_HOME}/target/:${STORAGE_HOME}/jars/dataclayclient.jar"
  export DATACLAYCLIENTCONFIG=${clientConf}

  #-----------------------------------------------------
  # SET APPLICATION EXECUTION VARIABLES
  #-----------------------------------------------------
  TEXTCOLALIAS="MyTextCol"
  TEXTFILE="/home/bsc19/bsc19533/Wordcount/data/file"
  NUMTEXTSPERFILE=4
  COUNTSPERTEXT=1
  WCOP=4 #4 #1
  RTOP=3 #3 #1


  #-----------------------------------------------------
  # SUBMIT
  #-----------------------------------------------------
  ENQUEUEFLAGS="--num_nodes=4 --tasks_per_node=4 --exec_time=10 --worker_working_dir=gpfs --network=infiniband"
  DEBUGFLAGS="--log_level=debug --tracing=false --graph=true"
  MEMFLAGS="--jvm_workers_opts=\"-Xms1024m,-Xmx8496m,-Xmn400m\""
  STORAGEFLAGS="--storage_home=${STORAGE_HOME} --storage_props=${propsFile} --task_execution=compss" # --task_execution=external

  enqueue_compss \
    --prolog="${APP_HOME}/scripts/application/registerApps.sh,$APPUSER,$DATASET,$stubsDir,$clientConf" \
    --prolog="${APP_HOME}/scripts/application/generateData.sh,$cp,$storageConf,$TEXTCOLALIAS,$TEXTFILE,$NUMTEXTSPERFILE" \
    --classpath=$cp \
    $ENQUEUEFLAGS \
    $DEBUGFLAGS \
    $MEMFLAGS \
    $STORAGEFLAGS \
    consumer.Wordcount $TEXTCOLALIAS -t $COUNTSPERTEXT -wcop $WCOP -rtop $RTOP

