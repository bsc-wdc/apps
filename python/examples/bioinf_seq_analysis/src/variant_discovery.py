#!/bin/python


from pycompss.api.api import compss_barrier
from pycompss.api.task import task
from pycompss.api.constraint import constraint


import os
from utils import runCMD

from config import *

SAMPLE_NUM = 0

@task(returns=str)
def haploCaller(bam):
    output_path = os.path.join(OUT_DIR, "output.g.vcf.gz_{0}".format(SAMPLE_NUM))
    cmd = [GATK, "HaplotypeCaller", "-R", REF, "-I", bam, "-O", output_path, "-ERC", "GVCF", "-L", "chr1", "-L", "chr2"]

    stdout = runCMD(cmd)

    return output_path

@task(returns=str)
def filterVCF(vcf):
    output_path = os.path.join(OUT_DIR, "filtered.vcf.gz_{0}".format(SAMPLE_NUM))
    cmd = [GATK, "VariantFiltration", "-R", REF, "-V", vcf, "-O", output_path]

    stdout = runCMD(cmd)

    return output_path


def discover_variants(bam, sample_number=0):
    global SAMPLE_NUM
    SAMPLE_NUM = sample_number

    ### variant discovery
    # variant calling: haplotype caller, mergeVCFs
    vcf = haploCaller(bam)
    
    # barrier + mergeVCF 
    # variant filtering: variant filteration
    filtered_vcf = filterVCF(vcf)

    return filtered_vcf
