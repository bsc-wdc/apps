#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)

  # Set common arguments
  MSIZE=4
  BSIZE=8
  MKL_THREADS=4
  VERIFY_OUTPUT=False
  tracing=false

  # Set arguments
  appArgs="${MSIZE} ${BSIZE} ${MKL_THREADS} ${VERIFY_OUTPUT}"

  # Environment variables
  ComputingUnits=4

  # Execute specific version launch
  cd base
  ./run_local.sh $tracing $ComputingUnits $appArgs
  cd ..