#!/bin/bash

  scriptDir=$(dirname $0)

  ##########################################
  # FIRST LEVEL FUNCTIONS
  ##########################################
  wait_and_get_jobId() {
    # Wait
    sleep 5

    # Get jobID
    jobDep=$(bjobs | tail -n 1 | cut -c -7)
    #echo "jobID = $jobDep"
  }

  ################
  #     MAIN     #
  ################

  computingUnits=(4 4 4 4 4)
  tasksPerNode=(16 16 16 16 16)
  coresMKL=(16 16 16 16 16)
  workers=(1 1 1 1 1)
  blockSize=(8192 8192 8192 8192 8192)
  matSize=(4 4 4 4 4)
  timeLim=(10 10 10 10 10)

  computingUnits=(8)
  tasksPerNode=(32)
  coresMKL=(16)
  workers=(1)
  blockSize=(8192)
  matSize=(4)
  timeLim=(30)

  jobDep="None"
  numTests=${#computingUnits[@]}
  for (( i=0; i<${numTests}; i++ ));
  do
      ${scriptDir}/launch.sh $jobDep $((workers[$i]+1)) $((timeLim[$i])) $((tasksPerNode[$i])) true $((matSize[$i])) $((blockSize[$i])) $((computingUnits[$i])) $((coresMKL[$i]))
      wait_and_get_jobId
      echo "##################PARAMS exec"$(echo $i)" ####################"
      echo ${workers[$i]}
      echo nodeCount=$((workers[$i]+1))
      echo matSize=$((matSize[$i]))
      echo blockSize=$((blockSize[$i]))
      echo computingUnits=$((computingUnits[$i]))
      echo coresMKL=$((coresMKL[$i]))
  done
