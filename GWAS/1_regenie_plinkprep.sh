#!./bin/bash
set -o pipefail 
set -o errexit

plink \
    --bed "${bedfile}" \
    --bim "${bimfile}" \
    --fam "${famfile}" \
    --geno 0.01 \
    --hwe 1e-15 \
    --indep-pairwise 1000 100 0.8 \
    --write-snplist \
    --out qc_ldpruned_snps_chr"${chr}"

export output=[ "qc_ldpruned_snps_chr${chr}.snplist" "qc_ldpruned_snps_chr${chr}.prune.in" "qc_ldpruned_snps_chr${chr}.prune.out" ]
mv ${output} ${OUTPUT_PATH}