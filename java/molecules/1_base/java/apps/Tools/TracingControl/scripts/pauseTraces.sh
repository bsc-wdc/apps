#!/bin/bash

WASABIJARS=$HOME/wasabidemo/jars
COMPSSMASTER=$HOME/COMPSs-SCO
TRACINGCONTROL=$COMPSSMASTER/Tools/TracingControl/lib
export WASABICLIENTCONFIG=$COMPSSMASTER/wasabi/cfgfiles/client.properties

echo ""
java -cp $TRACINGCONTROL/tracingcontrol.jar:$WASABIJARS/wasabiclient.jar severo.tracingcontrol.TracingControl pause 
echo "** Traces paused"
echo ""
