This details the colocalisation analysis between traits in preparation for Mendelian randomisation.

Burgess et al, 2023 details colocalisation as a sensitivity analysis but here it is applied as an instrument selection approach.

This is because genetic colocalisation of a locus between 2 traits is necessary for causality and specifically it attempts to distinguish between
1. Causal variant for trait A at locus X == causal variant for trait B at locus X i.e shared
OR
2. Distinct causal variants at locus X


The coloc package in R is used with SuSIE as a means of finemapping beforehand.
This entire process takes in the raw summary statistics for both traits A and B without filtering of SNPs post-GWAS (hence is a parallel approach to the FINEMAP/V2G via OTG pipeline).