#!/bin/python


from pycompss.api.api import compss_barrier
from pycompss.api.task import task
from pycompss.api.constraint import constraint


import os
from utils import runCMD

from config import *


@task(returns=str)
def haploCaller(bam, SAMPLE_NUM):
    output_path = os.path.join(OUT_DIR, "output.g.vcf.gz_{0}".format(SAMPLE_NUM))
    cmd = [GATK, "HaplotypeCaller", "-R", REF, "-I", bam, "-O", output_path, "-ERC", "GVCF", "-L", "chr1", "-L", "chr2"]

    stdout = runCMD(cmd)

    return output_path

@task(returns=str)
def filterVCF(vcf, SAMPLE_NUM):
    output_path = os.path.join(OUT_DIR, "filtered.vcf.gz_{0}".format(SAMPLE_NUM))
    cmd = [GATK, "VariantFiltration", "-R", REF, "-V", vcf, "-O", output_path]

    stdout = runCMD(cmd)

    return output_path
