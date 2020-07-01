#!/bin/python
from os.path import join

BASE_DIR="/path/to/base/dir"


TOOLS_DIR=join(BASE_DIR, "tools")
IN_DIR=join(BASE_DIR, "data/in")
OUT_DIR=join(BASE_DIR, "data/out")
META_DIR=join(BASE_DIR, "data/meta")


####################################################
########### TOOLS DIR 
####################################################

PICARD=join(TOOLS_DIR, "picard.jar")
BWA=join(TOOLS_DIR, "bwa/bwa")
SAMTOOLS=(TOOLS_DIR, "samtools-1.10/samtools")
GATK=(TOOLS_DIR, "gatk-4.1.8.0/gatk")


####################################################
########### META DIR 
####################################################

REF=join(META_DIR, "resources_broad_hg38_v0_Homo_sapiens_assembly38.fasta")
DBSNP=join(META_DIR, "resources_broad_hg38_v0_Homo_sapiens_assembly38.dbsnp138.vcf")
