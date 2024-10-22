#!/bin/bash
set -o pipefail 
set -o errexit

#This defines the actual bed_prefix, assuming localisation of the input bed/bim/fam files

echo "${bedfile}"
echo "${bimfile}"
echo "${famfile}"

bed_prefix=/mnt/data/input/gs/fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/plink_bed/acaf_threshold.chr"${chrom}" 

regenie \
    --step 1 \
    --bed "${bed_prefix}" \
    --phenoFile "${pheno_file}" \
    --phenoCol "${pheno}" \
    --covarFile "${cov_file}" \
    --catCovarList sex \
    --covarColList "age,ht,wt,pc1,pc2,pc3,pc4,pc5,pc6,pc7,pc8,pc9,pc10" \
    --bsize 1000 \
    --extract "${step1_snplist}" \
    --verbose \
    --"${trait}" \ 
    --ref-first \
    --out "${pheno}"_step1_chr"${chrom}"

#regenie pt 2
regenie \
    --step 2 \
    --bed "${bed_prefix}" \
    --phenoFile "${pheno_file}" \
    --phenoCol "${pheno}" \
    --covarFile "${cov_file}" \
    --catCovarList sex \
    --covarColList "age,ht,wt,pc1,pc2,pc3,pc4,pc5,pc6,pc7,pc8,pc9,pc10" \
    --pred "${pheno}"_step1_chr"${chrom}"_pred.list \
    --bsize 1000 \
    --minMAC 50 \
    --verbose \
    --"${trait}" \
    --ref-first \
    --out "${pheno}"_step2_chr"${chrom}"

export regenie_results=${pheno}_step2_chr"{chrom}".regenie
mv ${regenie_results} -t ${OUTPUT_PATH}