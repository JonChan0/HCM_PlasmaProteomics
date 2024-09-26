version 1.0
#This runs a GWAS on a per-chromosome basis in the tasks
#The workflow scatters over all the chromsomes

task plink_prep_step1{ #This performs QC and extracts the LD-pruned SNPs
    input{
        File bed_file
        File bim_file
        File fam_file
    }

    command <<<
        set -uexo pipefail

        prefix=$(basename "~{bed_file}" ".bed" )
        chr=${cut "$prefix", -d "." -f 2}

        plink --bfile "$prefix" \
        --maf 0.01 \
        --mac 100 \
        --geno 0.01 \
        --hwe 1e-15 \
        --indep-pairwise 1000 100 0.8 \
        --write-snplist \
        --out qc_ldpruned_snps_"$chr"

    >>>

    output{
        File snplist_file = glob("*.snplist")[0]
    }

    runtime{
        docker: "us.gcr.io/broad-dsp-gcr-public/terra-jupyter-aou:2.2.14" #This is the Docker image containing plink
    }


}

task regenie_step1{ #This runs the first step of the regenie analysis i.e ridge regression
    input{
        File bed_file
        File bim_file
        File fam_file
        File snplist_file
        File pheno_file
        File covar_file
        String pheno_of_choice #This defines the phenotype to run the GWAS on
    }

    command <<<
        set -uexo pipefail

        prefix=$(basename "~{bed_file}" ".bed" )
        chr=${cut "$prefix", -d "." -f 2}

        regenie --step 1 \
        --bed "$prefix" \
        --bsize 1000 \
        --phenoFile "~{pheno_file}" \
        --covarFile "~{covar_file}" \
        --apply-rint \
        --phenoCol "~{pheno_of_choice}" \
        --covarColList "sex, ht, wt, pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9, pc10, age_""~{pheno_of_choice}" \
        --extract "~{snplist_file}" \
        --threads 4 \
        --out "~{pheno_of_choice}"_"$chr"_step1

    >>>

    output{
        File step1_out = glob("*step1*")[0]
    }

    runtime{
        docker: "us-central1-docker.pkg.dev/all-of-us-rw-prod/aou-rw-gar-remote-repo-docker-prod/skoyamamd/regenie_3.4.1:latest"
        cpu: 4
        
    }

}

task regenie_step2{ #This runs the second step i.e linear regression to test individual SNPs
    input{
        File bed_file
        File bim_file
        File fam_file
        File pheno_file
        File covar_file
        File step1_out
        String pheno_of_choice
    }

    command <<<
        set -uexo pipefail

        prefix=$(basename "~{bed_file}" ".bed" )
        chr=${cut "$prefix", -d "." -f 2}

        regenie --step 2 \
        --bed "$prefix" \
        --covarFile "~{covar_file}" \
        --phenoFile "~{pheno_file}" \
        --minMAC 50 \
        --bsize 1000 \
        --apply-rint \
        --pred "~{step1_out}" \
        --out "~{pheno_of_choice}"_"$chr"_step2
        --threads 4

    >>>

    output{
        File step2_out = glob("*step2*")[0]
    }

    runtime{
        docker: "us-central1-docker.pkg.dev/all-of-us-rw-prod/aou-rw-gar-remote-repo-docker-prod/skoyamamd/regenie_3.4.1:latest"
        cpu: 4
    }
}


workflow regenie_aous{ #This runs the entire workflow for all the chromosomes
    input{
        Array[File] bed_files
        Array[File] bim_files
        Array[File] fam_files
        File pheno_file
        File covar_file
        String pheno_of_choice
    }

    scatter (idx in range(length(bed_files))){ #This scatters over all the chromosomes but scatters across all 3 arrays simultaneously
        call plink_prep_step1{
            input: bed_file = bed_files[idx],
                   bim_file = bim_files[idx],
                   fam_file = fam_files[idx]
        }


        call regenie_step1{
            input: bed_file = bed_files[idx],
                   bim_file = bim_files[idx],
                   fam_file = fam_files[idx],
                   snplist_file = plink_prep_step1.snplist_file,
                   pheno_file = pheno_file,
                   covar_file = covar_file,
                   pheno_of_choice = pheno_of_choice
        }

        call regenie_step2{
            input: bed_file = bed_files[idx],
                   bim_file = bim_files[idx],
                   fam_file = fam_files[idx],
                   pheno_file = pheno_file,
                   covar_file = covar_file,
                   step1_out = regenie_step1.step1_out,
                   pheno_of_choice = pheno_of_choice
        }

    }
}



