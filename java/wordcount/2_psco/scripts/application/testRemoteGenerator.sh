#!/bin/bash

TEXTCOLALIAS="MyTextCol"

echo ""
echo "Executing:"
echo ""
echo "java -cp stubs:bin:../../lib/dataclayclient.jar producer.RemoteTextCollectionGenerator "../../cfgfiles/config.properties" $TEXTCOLALIAS $@"
echo ""
java -cp stubs:bin:../../lib/dataclayclient.jar producer.RemoteTextCollectionGenerator "../../cfgfiles//config.properties" $TEXTCOLALIAS $@
