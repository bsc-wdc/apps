#!/bin/bash
EXECUTION_DIR=~/.COMPSs/$SLURM_JOB_ID/storage
export DATACLAYCLIENTCONFIG=$EXECUTION_DIR/cfgfiles/client.properties
STUBS_PATH=$EXECUTION_DIR/stubs
APP_PATH=/gpfs/home/bsc19/bsc19234/STORAGE/GENERIC/kmeans/src
DATASET=hpc_dataset
USERNAME=bsc_user
PASSWORD=bsc_user

echo
echo "***"
echo "*** Creating account \`${USERNAME}\`"
echo "***"
$DATACLAY_TOOL NewAccount $USERNAME $PASSWORD

echo
echo "***"
echo "*** Creating a new DataContract called \`${DATASET}\`"
echo "***"
$DATACLAY_TOOL NewDataContract $USERNAME $PASSWORD $DATASET $USERNAME

echo
echo "***"
echo "*** Registering all the data model \`classes\` in \`./stub\`"
echo "***"
$DATACLAY_TOOL NewModel $USERNAME $PASSWORD model $APP_PATH/model python

echo
echo "***"
echo "*** Getting stubs and putting them in $STUBS_PATH"
echo "***"
rm -rf $STUBS_PATH
mkdir -p $STUBS_PATH
$DATACLAY_TOOL GetStubs $USERNAME $PASSWORD model $STUBS_PATH
