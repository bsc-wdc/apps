 #!/bin/bash

# CHECK PATH: java, ant, ajc ...
DCBASEDIR=/apps/COMPSs/DATACLAY
export ANT_HOME=$DCBASEDIR/ant/apache-ant-1.9.6/
export ASPECTJ_HOME=$DCBASEDIR/Packages/aspectj
export JAVA_HOME=$DCBASEDIR/Packages/JDK
export PATH=$JAVA_HOME/bin:$ASPECTJ_HOME/bin:$ANT_HOME/bin:$PATH
echo "JAVA and AJC VERSION"
java -version 2>&1
ajc -version 2>&1

# CHECK ARGUMENTS
if [ $# -ne 4 ]; then
	echo "Bad arguments. Usage: $0 <AppUser> <DatasetName> <StubsPath> <ClientConfigPath>"
	exit -1
fi
APPSUSER=$1
DATASET=$2
STUBSPATH=$3
DATACLAYCLIENTCONFIG=$4
export DATACLAYCLIENTCONFIG=$DATACLAYCLIENTCONFIG
chmod 777 $DATACLAYCLIENTCONFIG

echo "--------- REGISTER_APPS ARGS -----------------"
echo " APPSUSER             $APPSUSER"
echo " DATASET              $DATASET"
echo " STUBSPATH            $STUBSPATH"
echo " DATACLAYCLIENTCONFIG $DATACLAYCLIENTCONFIG"
echo "----------------------------------------------"

APPSDIR="$HOME"
REGISTRATOR="Registrator"
NAMESPACE="WordcountNS"
CLASSNAME="model.TextCollectionIndex"
BINPATH="$APPSDIR/Wordcount/target"

BASETOOL="java -cp $DCBASEDIR/tools/lib/dataClayTools.jar:$DCBASEDIR/jars/dataclayclient.jar"
$BASETOOL tools.AccountRegistrator admin admin $REGISTRATOR $REGISTRATOR
$BASETOOL tools.AccountRegistrator admin admin $APPSUSER $APPSUSER

#$BASETOOL tools.RegisterToDataclayPublicContract $APPSUSER $APPSUSER
$BASETOOL tools.ClassRegistrator $NAMESPACE $CLASSNAME $BINPATH $REGISTRATOR $REGISTRATOR
$BASETOOL tools.ContractRegistrator $NAMESPACE $REGISTRATOR $REGISTRATOR $APPSUSER
$BASETOOL tools.DataContractRegistrator $DATASET $APPSUSER $APPSUSER $APPSUSER

mkdir -p $STUBSPATH
$BASETOOL tools.GetStubs $APPSUSER $APPSUSER $STUBSPATH
