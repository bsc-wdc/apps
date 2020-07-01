#!/bin/python

from pycompss.api.api import compss_barrier
from pycompss.api.task import task
from pycompss.api.constraint import constraint

import os
from utils import runCMD


from config import *

SAMPLE_NUM = 0

@task(returns=str)
def annotateVCF(bam, vcf):
    output_path = os.path.join(OUT_DIR, "annotated.vcf_{0}".format(SAMPLE_NUM))

    cmd = [GATK, "VariantAnnotator", "-I", bam, "-R", REF, "-V", vcf, "-O", output_path, "-A", "Coverage", "--dbsnp", DBSNP]

    stdout = runCMD(cmd)

    return output_path


def annotate_variants(bam, vcf, sample_number=0):
    global SAMPLE_NUM
    SAMPLE_NUM = sample_number
    
    return annotateVCF(bam, vcf)
