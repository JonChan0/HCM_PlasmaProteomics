#!/bin/bash
set -o pipefail 
set -o errexit

awk -v var="${pheno}" -F'\t' 'NR==1 {for (i=1; i<=NF; i++) if ($i == var) col=i} NR > 1 && $col != "NA" {print $1, $2}' "${pheno_file}" > nonNA_plinkids.txt

plink \
    --bed "${bedfile}" \
    --bim "${bimfile}" \
    --fam "${famfile}" \
    --keep nonNA_plinkids.txt \
    --extract "${step1_snplist}" \
    --write-snplist \
    --mac 50 \
    --memory 14000 \
    --out "${pheno}"_qc_ldpruned_MAC50filtered_chr"${chr}"

export output_snplist="${pheno}_qc_ldpruned_MAC50filtered_chr${chr}.snplist"
mv ${output_snplist} -t ${OUTPUT_PATH}