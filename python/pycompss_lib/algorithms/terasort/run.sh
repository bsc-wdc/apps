#!/bin/bash -e

  # Define script directory for relative calls
  # scriptDir=$(pwd)/$(dirname $0)

  # Set common arguments
  numFragments=10
  numEntries=100
  tracing=false

  # Set arguments
  appArgs="${numFragments} ${numEntries}"

  # Execute specific version launch
  1_base/run.sh $tracing $appArgs
  # 2_base_oo/run.sh $tracing $appArgs
