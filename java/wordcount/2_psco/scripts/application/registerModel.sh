#!/bin/bash

CONSUMER="Consumer"
REGISTRATOR="Registrator"
NAMESPACE="WordcountNS"
CLASSNAME="model.TextCollectionIndex"
DATASET="WordcountDS"
BINPATH="bin"
STUBSPATH="stubs"
mkdir -p $STUBSPATH

TOOLSLAUNCHER="java -cp bin:lib/dataclayclient.jar"

# Build model to register
javac -cp $STUBSPATH:./lib/dataclayclient.jar src/model/*.java -d $BINPATH

# Register accounts if necessary
$TOOLSLAUNCHER tools.AccountRegistrator admin admin $REGISTRATOR $REGISTRATOR 2>/dev/null
$TOOLSLAUNCHER tools.AccountRegistrator admin admin $CONSUMER $CONSUMER 2>/dev/null


# Register classes and model contract
$TOOLSLAUNCHER tools.ClassRegistrator $NAMESPACE $CLASSNAME $BINPATH $REGISTRATOR $REGISTRATOR 2>/dev/null
$TOOLSLAUNCHER tools.ContractRegistrator $NAMESPACE $REGISTRATOR $REGISTRATOR $CONSUMER 2>/dev/null

# Download stubs
$TOOLSLAUNCHER tools.GetStubs $CONSUMER $CONSUMER $STUBSPATH 2>/dev/null

# Register dataset and data contract
$TOOLSLAUNCHER tools.DataContractRegistrator $DATASET $CONSUMER $CONSUMER $CONSUMER 2>/dev/null

