#!/bin/bash

set -o pipefail 
set -o errexit

#This imports in the actual plink genetic files
echo "${bedfile}"
echo "${bimfile}"
echo "${famfile}"

regenie \
    --step 1 \
    --bed "${bed_prefix}" \
    --phenoFile "${pheno_file}" \
    --phenoCol "${pheno}" \
    --covarFile "${cov_file}" \
    --catCovarList sex \
    --covarColList ht, wt, pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9, pc10, age_"${pheno}" \
    --bsize 1000 \
    --extract "${step1_snplist}" \
    --verbose \
    --"${trait}" \ 
    --apply-rint \
    --ref-first \
    --out "${prefix}"_step1_chr"${chrom}"

#regenie pt 2
regenie \
    --step 2 \
    --bed "${bed_prefix}" \
    --phenoFile "${pheno_file}" \
    --phenoCol "${pheno}" \
    --covarFile "${cov_file}" \
    --catCovarList sex \
    --covarColList ht, wt, pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9, pc10, age_"${pheno}" \
    --pred "${prefix}"_step1_chr"${chrom}"_pred.list \
    --minMAC 50 \
    --bsize 1000 \
    --verbose \
    --"${trait}" \
    --apply-rint \
    --ref-first \
    --out "${prefix}"_step2_chr"${chrom}"

export regenie_results="${prefix}_step2_chr${chrom}_${phen_col}.regenie"
echo "regenie_results: ${regenie_results}"
mv ${regenie_results} ${OUTPUT_PATH}