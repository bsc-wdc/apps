#!/bin/bash
#BSUB -J KMeansCOMPSsDC
#BSUB -oo compss-%J.out
#BSUB -eo compss-%J.err
#BSUB -n 4 
#BSUB -R "span[ptile=1]"
#BSUB -W 0:30
#BSUB -q bsc_cs 

# --------------------------------------
# -     COMMON VARIABLES               -
# --------------------------------------
DCBASEDIR="/apps/COMPSs/DATACLAY"
HOSTSARRAY=($LSB_HOSTS)     
CLIENTHOST=${HOSTSARRAY[0]} #1st node for client 
LMNODE=${HOSTSARRAY[1]}     #2nd node for LM
DSNODES=${HOSTSARRAY[@]:2}  #rest of nodes for DS
echo APPNODE: $CLIENTHOST
echo LMNODE:  $LMNODE
echo DSNODES: $DSNODES


# --------------------------------------
# -     APP VARIABLES                  -
# --------------------------------------
APPSDIR="$HOME"
CURAPP="KMeans"
APPSUSER="Consumer"
DATASET="${CURAPP}DS"
FRAGMENTCOLALIAS="mykmeans"
NUMFRAGMENTS=3
VECTORSPERFRAGMENT=3
DIMENSIONSPERVECTOR=2
KVAR=2
ITERATIONS=2
DODEBUG="-debug"


# --------------------------------------
# -       GENERATE CONF FILES          -
# --------------------------------------
JOBDIR="$HOME/$CURAPP/$LSB_JOBID"; mkdir -p $JOBDIR/cfgfiles
STUBSPATH="$JOBDIR/stubs"
STORAGEPROPERTIES="$JOBDIR/storage.properties"
$DCBASEDIR/scripts/generateStorageConf.sh --jobdir $JOBDIR --lmnode $LMNODE --networksuffix "-ib0" --account $APPSUSER --stubsdir $STUBSPATH --dataset $DATASET > $STORAGEPROPERTIES
chmod 777 $STORAGEPROPERTIES



# --------------------------------------
# -       EXECUTE                      -
# --------------------------------------
# Start DataClay
$DCBASEDIR/scripts/_startDataClay.sh --lmnode $LMNODE --dsnodes "$DSNODES" --dcdir $DCBASEDIR --jobid $LSB_JOBID --networksuffix "-ib0" &
sleep 45


# Register model and contracts
DATACLAYCLIENTCONFIG="$JOBDIR/cfgfiles/client.properties"
$HOME/$CURAPP/registerModel.sh $APPSUSER $DATASET $STUBSPATH $DATACLAYCLIENTCONFIG



# Run Generator
CLASSPATHFLAG="$STUBSPATH:${APPSDIR}/$CURAPP/bin/:$DCBASEDIR/jars/dataclayclient.jar"
java -cp $CLASSPATHFLAG producer.FragmentDataClayGenerator $STORAGEPROPERTIES $FRAGMENTCOLALIAS $NUMFRAGMENTS $VECTORSPERFRAGMENT $DIMENSIONSPERVECTOR 

# Run KMeans with COMPSs
ENVFLAGS="--worker_working_dir=${JOBDIR} --network=infiniband"
#DEBUGFLAGS="--log_level=off --tracing=false --graph=false"
DEBUGFLAGS="--summary --log_level=debug --tracing=false --graph=false"
MEMFLAGS="--jvm_workers_opts=\"-Xms1024m,-Xmx8496m,-Xmn400m\""
STORAGEFLAGS="--storage_conf=$STORAGEPROPERTIES --task_execution=compss" 

launch_compss --master_node=$CLIENTHOST --worker_nodes="$DSNODES" --classpath=$CLASSPATHFLAG $ENVFLAGS $DEBUGFLAGS $MEMFLAGS $STORAGEFLAGS consumer.KMeans $FRAGMENTCOLALIAS -k $KVAR -iterations $ITERATIONS $DODEBUG

# Stop DataClay (optional) 
$DCBASEDIR/scripts/_stopDataClay.sh --lmnode $LMNODE --dsnodes "$DSNODES" --dcdir $DCBASEDIR --jobid $LSB_JOBID --networksuffix "-ib0"

wait

