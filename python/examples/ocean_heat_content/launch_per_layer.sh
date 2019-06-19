#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  numNodes=2
  executionTime=20

    # Execute specific version launch
  ${scriptDir}/per_layer/launch.sh $numNodes $executionTime \
                                   /gpfs/projects/bsc19/COMPSs_DATASETS/oceanheatcontent/all/ \
                                   /gpfs/projects/bsc32/cs_collaboration/numba_ohc/dataset/mesh_mask_nemo.Ec3.2_O1L75.nc \
                                   /gpfs/projects/bsc32/cs_collaboration/numba_ohc/dataset/mask.regions.Ec3.2_O1L75.nc \
                                   /gpfs/projects/bsc19/bsc19234/EARTH/ocean_heat_content/results/
