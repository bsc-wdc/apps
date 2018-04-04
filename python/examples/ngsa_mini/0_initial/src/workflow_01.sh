#! /bin/sh
#
# Perform mapping and split the results

BIN_DIR=`dirname $0`
BWA_BIN=$BIN_DIR/bwa
SSC_BIN=$BIN_DIR/splitSam2Contig2
N_THREADS=1

INPUT_DIR=$1
MED_DIR=$2
BWA_FILE=$3
CONTIG_FILE=$4

EXTS=()
for file in ${INPUT_DIR}/* ; do
  extension=${file##*.}
  EXTS+=($extension)
done
EXTS=(`for ext in "${EXTS[@]}"; do echo $ext; done | sort | uniq`)

for ext in "${EXTS[@]}"; do
  files=(`ls ${INPUT_DIR}/*.${ext}`)
  SEQ1_FILE=${files[0]}
  SEQ2_FILE=${files[1]}

  ALN_DIR=$MED_DIR/$ext
  mkdir -p $ALN_DIR

  ALN_PREF="${BWA_BIN} aln -t ${N_THREADS} ${BWA_FILE}"

  SEQ1_SAI=$ALN_DIR/`basename ${SEQ1_FILE}`.sai
  CMD="${ALN_PREF} ${SEQ1_FILE} > ${SEQ1_SAI}"
  eval $CMD

  SEQ2_SAI=$ALN_DIR/`basename ${SEQ2_FILE}`.sai
  CMD="${ALN_PREF} ${SEQ2_FILE} > ${SEQ2_SAI}"
  eval $CMD

  SAM_FILE=$ALN_DIR/0.sam
  CMD="${BWA_BIN} sampe ${BWA_FILE} ${SEQ1_SAI} ${SEQ2_SAI} ${SEQ1_FILE} ${SEQ2_FILE} > ${SAM_FILE}"
  eval $CMD

  CMD="${SSC_BIN} ${CONTIG_FILE} ${SAM_FILE} ${ALN_DIR} 1>&2"
  eval $CMD

  for path in ${ALN_DIR}/*.sam ; do
    if [ $path = $ALN_DIR/0.sam ]      || \
       [ $path = $ALN_DIR/single.sam ] || \
       [ $path = $ALN_DIR/unmap.sam ] ; then
      continue
    fi
    target=`basename $path`
    cat $path >> $MED_DIR/$target
  done
done
