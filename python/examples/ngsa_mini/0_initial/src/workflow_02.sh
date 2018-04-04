#! /bin/sh
#
# Perform merge - move to shared directory from local directory

BIN_DIR=`dirname $0`

ID=$1
TARGET_DIR=$2    # rank local directory
SHRD_TMP_DIR=$3  # rank shared directory

for sam_file in ${TARGET_DIR}/*.sam ; do
  target_dir=$SHRD_TMP_DIR/`basename ${sam_file}`
  mkdir -p $target_dir
  mv $sam_file $target_dir/$ID.sam
done
