#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(pwd)

  # Set common arguments
  numNodes=2
  executionTime=20

    # Execute specific version launch
  ${scriptDir}/per_layer/run_local.sh $numNodes $executionTime \
                                      /path/to/oceanheatcontent/dataset/ \
                                      /path/to/mesh_mask.nc \
                                      /path/to/mask_regions.nc \
                                      /path/to/results/
