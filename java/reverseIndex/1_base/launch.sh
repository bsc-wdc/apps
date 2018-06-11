#!/bin/bash -e

compile_and_execute() {
	mvn clean package
	runcompss $1 $2 ${@:3}
}

  # Define script directory for relative calls
  scriptDir=$(dirname $0)
  appName=$2
  runcompssOpts="$1 --classpath=${PWD}/lib/htmllexer-2.1.jar:${PWD}/lib/htmlparser-2.1.jar:${PWD}/reverse.jar"
  appArgs=${@:3}

  compile_and_execute  $runcompssOpts $appName $appArgs

  exit 0
