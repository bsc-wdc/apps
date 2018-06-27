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

  wait_and_get_jobID_MT() {
    sleep 5
    jobID=$(squeue | grep $(whoami) | sort -n - | tail -n 1 | awk '{print $1}')
    echo
  }

  ################
  #     MAIN     #
  ################

  computingUnits=(4 4 4 4 4)
  tasksPerNode=(16 16 16 16 16)
  coresMKL=(16 16 16 16 16)
  workers=(2 1 1 1 1)
  blockSize=(512 512 512 512 512)  
  #blockSize=(8192 8192 8192 8192 8192)
  matSize=(4 4 4 4 4)
  timeLim=(3 3 3 3 3)

  computingUnits=(1 1 1 1  1 1 1 1)
  tasksPerNode=(32 32 32 32  32 32 32 32)
  coresMKL=(1 1 1 1  1 1 1 1)
  workers=(7 3 7 15  1 3 6 12)
  blockSize=(4096 4096 4096  4096 4096 4096 4096)
  matSize=(16 16 16 16  48 144 288 576)
  timeLim=(60 60 60 60  60 60 60 60)

  jobDep="None"
  #numTests=${#workers[@]}
  numTests=1
  for (( i=0; i<${numTests}; i++ ));
  do
      ${scriptDir}/launch.sh $jobDep $((workers[$i]+1)) $((timeLim[0])) $((tasksPerNode[0])) true $((matSize[0])) $((blockSize[0])) $((computingUnits[0])) $((coresMKL[0]))
      wait_and_get_jobID_MT
      echo "##################PARAMS exec"$(echo $i)" ####################"
      echo ${workers[$i]}
      echo nodeCount=$((workers[$i]+1))
      echo matSize=$((matSize[0]))
      echo blockSize=$((blockSize[0]))
      echo computingUnits=$((computingUnits[0]))
      echo coresMKL=$((coresMKL[0]))
  done
