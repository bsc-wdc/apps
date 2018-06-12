 #!/bin/bash

# CHECK PATH: java, ant, ajc ...
DCBASEDIR=/apps/COMPSs/DATACLAY

# CHECK ARGUMENTS
if [ $# -ne 4 ]; then
	echo "Bad arguments. Usage: $0 <AppUser> <DatasetName> <StubsPath> <ClientConfigPath>"
	exit -1
fi

APPSUSER=$1
DATASET=$2
STUBSPATH=$3
export DATACLAYCLIENTCONFIG=$4 #exported for ClientManagementLib

APPSDIR="$HOME"
REGISTRATOR="Registrator"
NAMESPACE="KMeansNS"
CLASSNAME="model.FragmentCollection"
BINPATH="$APPSDIR/KMeans/bin"

BASETOOL="java -cp $DCBASEDIR/tools/lib/dataClayTools.jar:$DCBASEDIR/jars/dataclayadmin.jar"
$BASETOOL tools.AccountRegistrator admin admin $REGISTRATOR $REGISTRATOR
$BASETOOL tools.AccountRegistrator admin admin $APPSUSER $APPSUSER

#$BASETOOL tools.RegisterToDataclayPublicContract $APPSUSER $APPSUSER
#$BASETOOL tools.GetStubs $APPSUSER $APPSUSER $STUBSPATH

$BASETOOL tools.ClassRegistrator $NAMESPACE $CLASSNAME $BINPATH $REGISTRATOR $REGISTRATOR
$BASETOOL tools.ContractRegistrator $NAMESPACE $REGISTRATOR $REGISTRATOR $APPSUSER

mkdir -p $STUBSPATH
$BASETOOL tools.GetStubs $APPSUSER $APPSUSER $STUBSPATH $NAMESPACE

$BASETOOL tools.DataContractRegistrator $DATASET $APPSUSER $APPSUSER $APPSUSER
