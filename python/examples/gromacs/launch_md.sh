#!/bin/bash -e

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/src/lysozyme_in_water_@containerSingularity.py
  appClasspath=${scriptDir}/src/
  appPythonpath=${scriptDir}/src/

  # Retrieve arguments
  numNodes=2
  executionTime=$2
  tracing=$3

  # Leave application args on $@
  shift 3

  # Load necessary modules
  #module purge
  module load intel/2017.4 impi/2017.4 mkl/2017.4 bsc/1.0
  module load TrunkCAA
  JAVA_TOOL_OPTIONS=-Xss1280k
  #module load gromacs/2016.4   # exposes gmx_mpi binary
  # module load intel/2018.4 mkl/2018.4 impi/2018.4 gromacs/2018.3    # exposes gmx_mpi binary

  #export GMX_BIN=/home/nct00/nct00011/gromacs5.1.2/bin   # exposes gmx binary
  #module load fftw/3.3.6
  #export GMX_BIN=/home/nct00/nct00011/gromacs-2018.7/bin   # exposes gmx binary



  # Enqueue the application
  enqueue_compss \
    --qos=debug \
    #--reservation=PATC20-COMPSs \
    --num_nodes=$numNodes \
    --exec_time=$executionTime \
   # --master_working_dir=. \
    --worker_working_dir=scratch \
    --tracing=$tracing \
    --graph=true \
    -d \
    --classpath=$appClasspath \
    --pythonpath=$appPythonpath \
    --lang=python \
    $execFile $@

 
######################################################
# APPLICATION EXECUTION EXAMPLE
# Call:
#       ./launch_md.sh <NUMBER_OF_NODES> <EXECUTION_TIME> <TRACING> <CONFIG_PATH> <DATASET_PATH> <OUTPUT_PATH>
#
# Example:
#       ./launch_md.sh 2 10 true $(pwd)/config/ $(pwd)/dataset/ $(pwd)/output/
#
#####################################################
