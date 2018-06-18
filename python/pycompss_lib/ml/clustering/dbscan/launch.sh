#!/bin/bash -e

  # Define script directory for relative calls
  scriptDir=$(dirname $0)

  # Set common arguments
  numNodes=16
  tracing=false
  qos="debug"
  executionTime=120
  cpus_per_node=48
  worker_in_master_cpus=24
  scheduler="es.bsc.compss.scheduler.fifoDataScheduler.FIFODataScheduler"
  worker_working_dir=scratch

  # Set arguments:
  # Parameters: is_mn epsilon minPoints dataFile
  is_mn=true
  epsilon=0.015
  minPoints=10
  dataFile=3

  # Execute specifc version launch
  ${scriptDir}/base/launch.sh $numNodes $tracing $qos $executionTime $cpus_per_node $worker_in_master_cpus $scheduler $worker_working_dir $is_mn $epsilon $minPoints $dataFile
