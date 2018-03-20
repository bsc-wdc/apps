#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  jobDependency=None
  numNodes=2
  executionTime=5
  tasksPerNode=16
  tracing=false
  
  # Set arguments
  appArgs="4 2 1"
  # Arguments:
  #   <MSIZE> <BSIZE> <SEED>
  # where:
  #               * - MSIZE: Number of blocks of the matrix
  #               * - BSIZE: Number of elements per block
  #               * - SEED: Integer for random seed
 
  # Version used by default: random generated objects:
  #=== Version 2 ===
  #''Random Generation Objects'', where the matrices are randomly generated at execution time and stored internally as objects

  # Execute specifcversion launch  
  ${scriptDir}/1_base/launch.sh $jobDependency $numNodes $executionTime $tasksPerNode $tracing $appArgs



