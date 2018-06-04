#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # $1: Is used to execute the proper specific version, if the name of the version is 1_base, base must be inserted.
  # $2: First application argument 
  #  .
  #  .
  #  .
  # $N: Last application argument

  # Set common arguments
  tracing=false
  
  # Set arguments
  appArgs="2"
  # Arguments:
  #   <VALUE>
  # where:
  #               * - VALUE: Value to sum to our counter.

  runcompssOpts=" --tracing="$tracing""

  appDir=$(ls ${scriptDir} | grep "^[0-9]*[_]"$1"")

  # Execute specific version launch  
  ${scriptDir}/${appDir}/launch.sh $runcompssOpts $appArgs



