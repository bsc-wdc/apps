#!/bin/bash

java -Xmx4g -cp stubs:bin:../../lib/dataclayclient.jar producer.CleanCurrentTextCollection "../../cfgfiles/config.properties" "MyTextCol"
