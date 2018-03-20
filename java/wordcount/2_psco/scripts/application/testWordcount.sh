#!/bin/bash

TEXTCOLALIAS="MyTextCol"

echo ""
echo "Executing:"
echo ""
echo "java -cp stubs:bin:../../lib/dataclayclient.jar consumer.Wordcount $TEXTCOLALIAS $@"
echo ""

java -cp stubs:bin:../../lib/dataclayclient.jar consumer.Wordcount $TEXTCOLALIAS $@

