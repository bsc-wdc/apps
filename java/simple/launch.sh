#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  if [ "$#" -lt 1 ]; then
        echo "Usage: 
        # <VERSION>        : Is used to execute the proper specific version, if the name of the version is 1_base, base must be inserted.
	# <ARGS0-N>	   : Application arguments."

        exit 1
    
  fi  

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

  appDir=$(ls ${scriptDir} | grep "^[0-9]*[_]"$1"$") || true

  if [ -z "$appDir" ]; then
	echo "The version ""$1"" is not available.
We found these versions on the directory: 
	      "$(ls ${scriptDir} | grep "^[0-9]*[_][\H]*")""
	
	exit 1
  else
        # Execute specific version launch  
  	cd ${scriptDir}/${appDir}
	./launch.sh $runcompssOpts $appArgs
  fi
