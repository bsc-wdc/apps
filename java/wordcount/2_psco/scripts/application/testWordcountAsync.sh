#!/bin/bash

TEXTCOLALIAS="MyTextCol"

echo ""
echo "Executing:"
echo ""
echo "java -cp stubs:bin:../../lib/dataclayclient.jar consumer.WordcountAsync "../../cfgfiles/config.properties" $TEXTCOLALIAS $@"
echo ""

java -cp stubs:bin:../../lib/dataclayclient.jar consumer.WordcountAsync "../../cfgfiles/config.properties" $TEXTCOLALIAS $@
