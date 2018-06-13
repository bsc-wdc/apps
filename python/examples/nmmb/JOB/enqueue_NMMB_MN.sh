#!/bin/bash

  # Define NMMB Environment
  export PATH=$PATH:/gpfs/projects/bsc19/bsc19234/nmmb/MODEL-MN/exe

  export UMO_PATH=/gpfs/projects/bsc19/bsc19234/nmmb/
  export UMO_ROOT=$UMO_PATH/JOB

  export FIX=$UMO_PATH/PREPROC/FIXED
  export VRB=$UMO_PATH/PREPROC/VARIABLE
  export POST_CARBONO=$UMO_PATH/POSTPROC
  export SRCDIR=$UMO_PATH/MODEL-MN

  export OUTPUT=$UMO_PATH/PREPROC/output
  export OUTNMMB=$UMO_PATH/OUTPUT
  export UMO_OUT=$UMO_PATH/OUTPUT/CURRENT_RUN
 
  export GRB=$UMO_PATH/DATA/INITIAL
  export DATMOD=$UMO_PATH/DATA/STATIC
  export CHEMIC=
  export STE=
  export OUTGCHEM=
  export PREMEGAN=
  export TMP=/tmp

  export FNL=$GRB
  export GFS=$GRB

  # Define script variables
  scriptDir=$(pwd)/$(dirname $0)
  propertiesFile=${scriptDir}/nmmb_compss_MN.properties

  # Define NMMB environment constants
  export NEMS_NODES=4
  export NEMS_CUS_PER_NODE=16

  # Define MPI tricks
  export OMPI_MCA_coll_hcoll_enable=0
  export OMPI_MCA_mtl=^mxm

  # Enqueue
  jobDepFlag="--job_dependency=None --qos=debug"
  debugFlags="--log_level=off --summary"
  toolsFlags="--graph=true --tracing=false"

  enqueue_compss \
    --lang=python \
    --exec_time=30 \
    --num_nodes=5 \
    --max_tasks_per_node=16 \
    ${jobDepFlag} \
    --master_working_dir=. \
    --worker_working_dir=/gpfs/projects/bsc19/bsc19234/ \
    --network=infiniband \
    ${debugFlags} \
    ${toolsFlags} \
    --pythonpath=${scriptDir}/../src/nmmb \
    ${scriptDir}/../src/nmmb/Nmmb.py ${propertiesFile}

#     --worker_working_dir=scratch \