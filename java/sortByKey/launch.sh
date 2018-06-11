#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)
  
  if [ "$#" -lt 2 ]; then
	echo "Usage: 
	# <VERSION>	   : Is used to execute the proper specific version, if the name of the version is 1_base, base must be inserted.
  	# <IMPLEMENTATION> : Name of the kind of implementation inside a specific version, in_files, randobjects, randfiles are the possible options.
        # <ARGS0-N>	   : Application arguments."

	exit 1
		  
  fi

  # $1: Is used to execute the proper specific version, if the name of the version is 1_base, base must be inserted.
  # $2: Name of the kind of implementation inside a specific version
  # $3: First application argument 
  #  .
  #  .
  #  .
  # $N: Last application argument

  # Set common arguments
  tracing=false
  
  
  runcompssOpts=" --tracing="$tracing""

  appDir=$(ls ${scriptDir} | grep "^[0-9]*[_]"$1"$") || true
  appName="$2" 
  # Set app arguments

  appArgs=${@:3} 
  appName="$2"

  echo "Application arguments: "$appArgs""

    if [ -z "$appDir" ]; then
        echo "The version ""$1"" is not available.
We found these versions on the directory: 
              "$(ls ${scriptDir} | grep "^[0-9]*[_][\H]*")""

        exit 1
  else
        # Execute specific version launch  
	echo "Executing "$appImpl" implementation..."
	cd ${scriptDir}/${appDir}
        ./launch.sh $runcompssOpts $appName $appArgs
  fi 
