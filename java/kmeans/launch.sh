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
  
  # Arguments for in_files:
  #   <MSIZE> <BSIZE> <PATH_TO_DATASET_FOLDER>
  # where:
  #               * - MSIZE: Number of blocks of the matrix
  #               * - BSIZE: Number of elements per block
  #               * - PATH_TO_DATASET_FOLDER: Folder where matrices A and B are stored in files

  # Arguments for randobjects and randfiles:
  #   <MSIZE> <BSIZE> <SEED>
  # where:
  #               * - MSIZE: Number of blocks of the matrix
  #               * - BSIZE: Number of elements per block
  #               * - SEED: Integer for random seed
  
  runcompssOpts=" --tracing="$tracing""

  appDir=$(ls ${scriptDir} | grep "^[0-9]*[_]"$1"$") || true
  appName="$2" #Expected matmul.input.files.Matmul or matmul.randomGen.objects.Matmul or matmul.randomGen.files.Matmul
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
