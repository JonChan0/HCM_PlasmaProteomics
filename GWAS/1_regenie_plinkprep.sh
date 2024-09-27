#!./bin/bash

set -o pipefail 
set -o errexit

plink \
    --bed "${bedfile}" \
    --bim "${bimfile}" \
    --fam "${famfile}" \
    --maf 0.01 \
    --mac 100 \
    --geno 0.01 \
    --hwe 1e-15 \
    --indep-pairwise 1000 100 0.8 \
    --write-snplist \
    --out qc_ldpruned_snps_"${chr}"

export output="qc_ldpruned_snps_${chr}.snplist"
mv ${output} ${OUTPUT_PATH}