#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=src/kmeans.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  tracing=$1

  # Leave application args on $@
  shift 1

  # Generate dataset
  cd generator
  ./generateData.sh 16000 4 3 4
  cd ..
  if [ -d dataset ]; then
      rm -rf dataset
  fi
  mkdir dataset
  mv generator/*.txt dataset/.

  # Enqueue the application
  runcompss \
    --tracing=$tracing \
    --classpath=$appClasspath \
    --pythonpath=$appPythonpath \
    --lang=python \
    $execFile $@


######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./run_local.sh tracing dataset_path numV dim k
#
# Example:
#       ./run_local.sh false $(pwd)/dataset/ 16000 3 4
#