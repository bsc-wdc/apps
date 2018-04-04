#! /bin/sh
#
# Perform remove duplicates

BIN_DIR=`dirname $0`
SAMT_BIN=$BIN_DIR/samtools
N_MEMORY=800000000

TARGET_DIR=$1
REF_IDX_FILE=$2

for sam_file in ${TARGET_DIR}/*.sam ; do
  if [ `basename $sam_file` = "*.sam" ] ; then
    continue
  fi

  BASENAME=`basename $sam_file .sam`
  BAM_FILE=$TARGET_DIR/$BASENAME.bam
  CMD="$SAMT_BIN import $REF_IDX_FILE $sam_file $BAM_FILE 1>&2"
  eval $CMD

  S_BAM_FILE=$TARGET_DIR/$BASENAME.sort
  CMD="$SAMT_BIN sort -m $N_MEMORY $BAM_FILE $S_BAM_FILE 1>&2"
  eval $CMD

  RMD_FILE=$S_BAM_FILE.rmdup.bam
  CMD="$SAMT_BIN rmdup $S_BAM_FILE.bam $RMD_FILE 1>&2"
  eval $CMD
done
