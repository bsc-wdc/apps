#!/bin/bash -e

  # Define script directory for relative calls
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

  # Set common arguments
  jobDependency=${1:-None}
  numNodes=${2:-5} 
  cpusPerNode=${3:-16}
  executionTime=${4:-15}
  
  tracing=${5:-false}
  graph=${6:-true}
  logLevel=${7:-off}
  
  # Set application arguments
  # Arguments:
  #   <propertiesFile>
  # where:
  #             * - propertiesFile: Location of the NMMB properties
  #
  propertiesFile=${8:-${SCRIPT_DIR}/base/JOB/nmmb_compss_MN.properties}

  # Version used by default: base
  # Execute specifcversion launch  
  "${SCRIPT_DIR}"/base/launch.sh "${jobDependency}" "${numNodes}" "${cpusPerNode}" "${executionTime}" "${tracing}" "${graph}" "${logLevel}" "${propertiesFile}"

