#!/bin/bash

  # CHECK ARGUMENTS
  if [ $# -ne 5 ]; then
    echo "Bad arguments. Usage: $0 <cp> <storageConf> <TEXTCOLALIAS> <TEXTFILE> <NUMTEXTSPERFILE>"
    exit 1
  fi

  # Get arguments
  cp=$1
  storageConf=$2
  TEXTCOLALIAS=$3
  TEXTFILE=$4
  NUMTEXTSPERFILE=$5

  export DATACLAYCLIENTCONFIG=$storageConf
  chmod 777 $DATACLAYCLIENTCONFIG

  # Log Arguments
  echo "--------- GENERATE ARGS -----------------"
  echo " CP               $cp"
  echo " Storage Conf     $storageConf"
  echo " TextColAlias     $TEXTCOLALIAS"
  echo " TextFile         $TEXTFILE"
  echo " NumTextsPerFile  $NUMTEXTSPERFILE"
  echo "-----------------------------------------"

  # Execute data generation
  java -cp $cp \
     producer.RemoteTextCollectionGenerator \
      $storageConf \
      $TEXTCOLALIAS \
      $TEXTFILE \
      -t $NUMTEXTSPERFILE
  exitValue=$?
  
  # Exit value
  exit $exitValue

