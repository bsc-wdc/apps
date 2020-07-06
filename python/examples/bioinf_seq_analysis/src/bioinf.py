#!/bin/python

from data_cleanup import *
from variant_discovery import *
from variant_annotation import *
from utils import read


def gatk_best_practices_pipeline():
    ### raw samples
    ubams = read()

    print ubams
    for sample_number, ubam in enumerate(ubams):
        print ("Started processing: ", ubam)
          		
        ### Data Cleanup and preparation
        ### data cleanup
        ubam = sort(ubam, sample_number)
        usam = revertSAM(ubam, sample_number)
        fastq = convertSAMtoFASTQ(usam, sample_number)
    
        # map to reference: bwa, convert sam to bam, mergeBAMAlignment
        mapped_sam = bwa_map(fastq, sample_number)  
        mapped_bam = convertSAMtoBAM(mapped_sam, sample_number)
        sorted_bam = sort(mapped_bam, sample_number)
        merged_bam = mergeBamAlignment(sorted_bam, usam, sample_number)

        # mark duplicates: markDuplicates + sortsam
        marked_bam, dup_metrics = markDuplicates(merged_bam, sample_number)  
    
        # splitNCigarReads
        splitted_bam = splitNCigarReads(marked_bam, sample_number)
        output_rg = addOrReplaceReadGroups(splitted_bam, sample_number)
        
        # base recalibration: base recalibrator, apply recalibration, analyzeCovariates 
        recal_data = recalibrateBase(output_rg, sample_number)
        recalibrated_bam = applyBQSR(output_rg, recal_data, sample_number)
        recal_data_plot = analyzeCovariates(recal_data, sample_number)
    
        
        ### variant discovery
        # variant calling: haplotype caller, mergeVCFs
        vcf = haploCaller(recalibrated_bam, sample_number)
    
        # barrier + mergeVCF 
        # variant filtering: variant filteration
        filtered_vcf = filterVCF(vcf, sample_number)
    
        ### Variant Annotation 
        annotated_vcf = annotateVCF(mapped_bam, filtered_vcf, sample_number)


if __name__ == "__main__":
   gatk_best_practices_pipeline()


#enqueue_compss --qos=debug --num_nodes=2 --constraints=highmem --exec_time=30 --jvm_workers_opts="-Dcompss.worker.removeWD=false" --jvm_master_opts="-Dcompss.worker.removeWD=false" --worker_working_dir=/gpfs/scratch/bsc19/bsc19004/workspace/io_sched/bioinf/ --worker_in_master_cpus=0 --base_log_dir=/gpfs/scratch/bsc19/bsc19004/workspace/io_sched/bioinf/ -t -d -g bioinf.py
