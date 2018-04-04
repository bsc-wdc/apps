#! /bin/sh
#
# Perform analyze

BIN_DIR=`dirname $0`
SAMT_BIN=$BIN_DIR/samtools
SNP_BIN=$BIN_DIR/snp

INPUT_DIR=$1
OUT_DIR=$2
REF_FILE=$3

for sam_file in ${INPUT_DIR}/*.sam ; do
  if [ `basename $sam_file` = "*.sam" ] ; then
    continue
  fi

  BASENAME=`basename $sam_file .sam`
  RMD_FILE=$INPUT_DIR/$BASENAME.sort.rmdup.bam
  PILEUP_FILE=$INPUT_DIR/$BASENAME.pile
  CMD="$SAMT_BIN pileup -s -cf $REF_FILE $RMD_FILE > $PILEUP_FILE"
  eval $CMD

  OUT_FILE1=$OUT_DIR/$BASENAME.indel
  OUT_FILE2=$OUT_DIR/$BASENAME.snp
  OUT_FILE3=$OUT_DIR/$BASENAME.sum
  CMD="$SNP_BIN -INF $PILEUP_FILE -INDEL $OUT_FILE1 -SNP $OUT_FILE2 -SUM $OUT_FILE3 1>&2"
  eval $CMD
done
