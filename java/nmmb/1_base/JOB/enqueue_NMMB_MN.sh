#!/bin/bash

  # Define script directory for relative calls
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

  # Define NMMB Environment
  export PATH=$PATH:/gpfs/projects/bsc19/bsc19533/nmmb/MODEL-MN/exe

  export UMO_PATH=/gpfs/projects/bsc19/bsc19533/nmmb/
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

  # Retrieve arguments
  jobDependency=${1:-None}
  numNodes=${2:-5}
  cpusPerNode=${3:-16}
  executionTime=${4:-15}

  tracing=${5:-false}
  graph=${6:-true}
  logLevel=${7:-off}

  propertiesFile=${8:-${SCRIPT_DIR}/nmmb_compss_MN.properties}

  # Define NMMB.jar environment constants
  NEMS_NODES=$((numNodes - 1))
  export NEMS_NODES=${NEMS_NODES}
  export NEMS_CUS_PER_NODE=${cpusPerNode}

  # Define MPI tricks
  export OMPI_MCA_coll_hcoll_enable=0
  export OMPI_MCA_mtl=^mxm

  # Enqueue
  enqueue_compss \
    --num_nodes="${numNodes}" \
    --cpus_per_node="${cpusPerNode}" \
    --exec_time="${executionTime}" \
    --job_dependency="${jobDependency}" \
    \
    --classpath="${SCRIPT_DIR}"/../nmmb.jar
    --master_working_dir=. \
    --worker_working_dir=scratch \
    --network=infiniband \
    \
    --tracing="${tracing}" \
    --graph="${graph}" \
    --summary \
    --log_level="${logLevel}" \
    \
    nmmb.Nmmb "${propertiesFile}"

