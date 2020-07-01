#!/bin/python

from pycompss.api.api import compss_barrier
from pycompss.api.task import task
from pycompss.api.constraint import constraint

import os

from utils import runCMD
from config import *

SAMPLE_NUM = 0
 
@task(returns=str)
def revertSAM(ubam):
    output_path = os.path.join(OUT_DIR, "reverted.bam_{0}".format(SAMPLE_NUM))
    input_arg = "I={0}".format(ubam)
    output_arg = "O={0}".format(output_path)
    cmd = ["java", "-jar", PICARD, "RevertSam", input_arg, output_arg]

    stdout = runCMD(cmd)
    
    return output_path

@task(returns=str)
def convertSAMtoFASTQ(usam):
    output_path = os.path.join(OUT_DIR, "sample.fastq_{0}".format(SAMPLE_NUM))
    input_arg = "I={0}".format(usam)
    output_arg = "FASTQ={0}".format(output_path)
    cmd = ["java", "-jar", PICARD, "SamToFastq", input_arg, output_arg]

    stdout = runCMD(cmd)

    return output_path

@task(returns=str)
def bwa_map(fastq):
    output_path = os.path.join(OUT_DIR, "mapped.sam_{0}".format(SAMPLE_NUM))
    cmd = [BWA, "mem", REF, fastq]

    f = open(output_path, "w")
    stdout = runCMD(cmd, f)
    f.close()
 
    return output_path

@task(returns=str)
def convertSAMtoBAM(sam):
    output_path = os.path.join(OUT_DIR, "mapped.bam_{0}".format(SAMPLE_NUM))
    
    cmd = [SAMTOOLS, "view", "-bhS", sam]

    f = open(output_path, "w")
    stdout = runCMD(cmd, f)
    f.close()
 
    return output_path

@task(returns=str)
def sort(bam):
    output_path = os.path.join(OUT_DIR, "sorted.bam_{0}".format(SAMPLE_NUM))
    output_arg = "O={0}".format(output_path)
    input_arg = "I={0}".format(bam)
    sort_order = "SORT_ORDER=queryname"
    cmd = ["java", "-jar", PICARD, "SortSam", input_arg, output_arg, sort_order]

    stdout = runCMD(cmd)

    return output_path

@task(returns=str)
def mergeBamAlignment(mapped_bam, unmapped_sam):
    output_path = os.path.join(OUT_DIR, "merged.bam_{0}".format(SAMPLE_NUM))
    output_arg = "O={0}".format(output_path)
    input_arg1 = "ALIGNED={0}".format(mapped_bam)
    input_arg2 = "UNMAPPED={0}".format(unmapped_sam)
    ref = "R={0}".format(REF)
    
    cmd = ["java", "-jar", PICARD, "MergeBamAlignment", input_arg1, input_arg2, ref, output_arg]

    stdout = runCMD(cmd)

    return output_path

@task(returns=2)
def markDuplicates(mapped_bam):
    output_path1 = os.path.join(OUT_DIR, "marked_duplicates.bam_{0}".format(SAMPLE_NUM))
    output_arg = "O={0}".format(output_path1)
    output_path2 = os.path.join(OUT_DIR, "marked_dup_metrics")
    output_arg2 = "M={0}".format(output_path2)
    input_arg = "I={0}".format(mapped_bam)

    cmd = ["java", "-jar", PICARD, "MarkDuplicates", input_arg, output_arg, output_arg2]

    stdout = runCMD(cmd)

    return output_path1, output_path2

@task(returns=str)
def splitNCigarReads(bam):
    output_path = os.path.join(OUT_DIR, "splitted.bam_{0}".format(SAMPLE_NUM))
    cmd = [GATK, "SplitNCigarReads", "-R", REF, "-I", bam, "-O", output_path]

    stdout = runCMD(cmd)

    return output_path

@task(returns=str)
def addOrReplaceReadGroups(bam):
    output_path = os.path.join(OUT_DIR, "output_rg.bam_{0}".format(SAMPLE_NUM))
    output_arg = "O={0}".format(output_path)
    input_arg = "I={0}".format(bam)
    rgid = "RGID=4"
    rglb = "RGLB=lib1"
    rgpl = "RGPL=ILLUMINA"
    rgpu = "RGPU=unitl"
    rgsm = "RGSM=20"

    cmd = ["java", "-jar", PICARD, "AddOrReplaceReadGroups", input_arg, output_arg, rgid, rglb, rgpl, rgpu, rgsm]

    stdout = runCMD(cmd)

    return output_path

@task()
def indexBAM(bam):
    cmd = [SAMTOOLS, "index", bam]

    stdout = runCMD(cmd)

@task(returns=str)
def recalibrateBase(bam):
    output_path = os.path.join(OUT_DIR, "recal_data.table_{0}".format(SAMPLE_NUM))
    known_sites1 = DBSNP
  
    cmd = [GATK, "BaseRecalibrator", "-I", bam, "-R", REF, "--known-sites", known_sites1, "-O", output_path, "-L", "chr1", "-L", "chr2"]

    stdout = runCMD(cmd)

    return output_path

@task(returns=str)
def applyBQSR(bam, recal_data):
    output_path = os.path.join(OUT_DIR, "recalibrated.bam_{0}".format(SAMPLE_NUM))
    cmd = [GATK, "ApplyBQSR", "-R", REF, "-I", bam, "--add-output-sam-program-record", "--use-original-qualities", "-O", output_path, "--bqsr-recal-file", recal_data]

    stdout = runCMD(cmd)

    return output_path

@task(returns=str)
def analyzeCovariates(recal_data):
    output_path = os.path.join(OUT_DIR, "AnalyzeCovariates.pdf_{0}".format(SAMPLE_NUM))
    cmd = [GATK, "AnalyzeCovariates", "-bqsr", recal_data, "-plots", output_path]

    stdout = runCMD(cmd)

    return output_path


def clean_and_prepare(ubam, sample_number=0):
    global SAMPLE_NUM
    SAMPLE_NUM = sample_number

    ### data cleanup
    ubam = sort(ubam)
    usam = revertSAM(ubam)
    fastq = convertSAMtoFASTQ(usam)
    
    # map to reference: bwa, convert sam to bam, mergeBAMAlignment
    mapped_sam = bwa_map(fastq)  
    mapped_bam = convertSAMtoBAM(mapped_sam)
    sorted_bam = sort(mapped_bam)
    merged_bam = mergeBamAlignment(sorted_bam, usam)

    # mark duplicates: markDuplicates + sortsam
    marked_bam, dup_metrics = markDuplicates(merged_bam)  
    
    # splitNCigarReads
    splitted_bam = splitNCigarReads(marked_bam)
    output_rg = addOrReplaceReadGroups(splitted_bam)
    indexBAM(output_rg)
    
    compss_barrier()
    
    # base recalibration: base recalibrator, apply recalibration, analyzeCovariates 
    recal_data = recalibrateBase(output_rg)
    recalibrated_bam = applyBQSR(output_rg, recal_data)
    # NOTE: To execute analyzeCovariants, you should add Rscript directory
    # to your env $PATH and install required R dependencies
    #recal_data_plot = analyzeCovariates(recal_data)
    recal_data_plot = ""
    
    return recalibrated_bam, recal_data_plot
