#!/bin/bash -e

compile_and_execute() {
  err1=$({ cd ${scriptDir}/jar ;} 2>&1) || true
  err2=$({ cd ${scriptDir}/target ;} 2>&1) || true
  
  if [ -n "$err1" ] && [ -n "$err2" ]; then 
	echo "There is no ./jar or ./target folder, I am going to build it..."
	mvn clean package
	cd ${scriptDir}/target
	runcompss $1 $2 $3	
  elif [ -z "$err1" ]; then
	cd ${scriptDir}/jar
	runcompss $1 $2 $3
  elif [ -z "$err2" ]; then
	cd ${scriptDir}/target
        runcompss $1 $2 $3
  fi
}

  # Define script directory for relative calls
  scriptDir=$(dirname $0)
  appName="simple.Simple"
  runcompssOpts=$1
  appArgs=$2

  compile_and_execute  $runcompssOpts $appName $appArgs

  exit 0
