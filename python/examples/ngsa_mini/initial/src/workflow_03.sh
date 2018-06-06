#! /bin/sh
#
# Perform merge - move to local directory from shared directory

BIN_DIR=`dirname $0`

ID=$1
NPROCS=$2
TARGET_DIR=$3    # rank local directory
SHRD_TMP_DIR=$4  # rank shared directory

FILES=()
for file in $SHRD_TMP_DIR/* ; do
  FILES+=($file)
done
COUNT=${#FILES[*]}

if [ $ID -lt $COUNT ]; then
  I=$ID
  while [ $I -lt $COUNT ]; do
    file=${FILES[$I]}
    contig=`basename $file .sam`
    target_dir=$TARGET_DIR/$contig
    target_file=$TARGET_DIR/$contig.sam
    mv $file $target_dir 2>/dev/null
    if [ $? == 0 ]; then
      cat $target_dir/*.sam >> $target_file
      rm -fr $target_dir
    fi
    ((I=$I+$NPROCS))
  done
fi
