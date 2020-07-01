#!/bin/python

from data_cleanup import clean_and_prepare
from variant_discovery import discover_variants
from variant_annotation import annotate_variants
from utils import read, checkDirs


def gatk_best_practices_pipeline():
    checkDirs()

    ### raw samples
    ubams = read()

    sample_number = 0
    for ubam in ubams:
        print ("Started processing: ", ubam)
          		
        ### Data Cleanup and preparation
        recalibrated_bam, recal_data_plot = clean_and_prepare(ubam, sample_number)
    
        ### Variant Discovery
        filtered_vcf = discover_variants(recalibrated_bam, sample_number)
    
        ### Variant Annotation 
        annotated_vcf = annotate_variants(recalibrated_bam, filtered_vcf, sample_number)

        sample_number += 1


if __name__ == "__main__":
   gatk_best_practices_pipeline()
