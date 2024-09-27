#!/usr/bin/env python
# coding: utf-8

# # GWAS via REGENIE for PP
# 
# This uses dsub (as opposed to WDL + Cromwell) to submit bash scripts corresponding to REGENIE for GWAS.
# 
# Template for code: https://workbench.researchallofus.org/workspaces/aou-rw-5981f9dc/aouldlgwasregeniedsubctv6duplicate/analysis/preview/4.0_regenie_dsub_HP_TM.ipynb 
# i.e the GWAS for LDL-C done by Bick et al, 2024
# 
# Modified for my appplications.

# In[1]:


## Python Package Import
import sys
import os 
import numpy as np
import pandas as pd
from datetime import datetime

##Ensuring dsub is up to date
get_ipython().system('pip3 install --upgrade dsub')


# In[2]:


#Defining environment variables
# Save this Python variable as an environment variable so that its easier to use within %%bash cells.
get_ipython().run_line_magic('env', 'JOB_ID={LINE_COUNT_JOB_ID}')
## Defining necessary pathways
my_bucket = os.environ['WORKSPACE_BUCKET']
## Setting for running dsub jobs
pd.set_option('display.max_colwidth', 0)

USER_NAME = os.getenv('OWNER_EMAIL').split('@')[0].replace('.','-')

# Save this Python variable as an environment variable so that its easier to use within %%bash cells.
get_ipython().run_line_magic('env', 'USER_NAME={USER_NAME}')


# ## Setting Variables for dsub Job
# 
# This requires modification for each different phenotype you run.

# In[3]:


## MODIFY FOR FULL DATA RUN 
JOB_NAME='REGENIE_ntprobnp'

# Save this Python variable as an environment variable so that its easier to use within %%bash cells.
get_ipython().run_line_magic('env', 'JOB_NAME={JOB_NAME}')


# In[4]:


## Analysis Results Folder 
line_count_results_folder = os.path.join(
    os.getenv('WORKSPACE_BUCKET'),
    'dsub',
    'results',
    JOB_NAME,
    USER_NAME,
    datetime.now().strftime('%Y%m%d'))

line_count_results_folder


# In[6]:


## Where the output files will go
output_files = os.path.join(line_count_results_folder, "results")
print(output_files)

OUTPUT_FILES = output_files

# Save this Python variable as an environment variable so that its easier to use within %%bash cells.
get_ipython().run_line_magic('env', 'OUTPUT_FILES={OUTPUT_FILES}')


# ## REGENIE Bash Script
# 
# This details and writes out a .sh script in the local Jupyter disk and then uploads it to GCP Bucket in order for dsub to run it.

# In[22]:


#This is the plink preparatory in REGENIE 
filename='1_regenie_plinkprep.sh'

script = '''
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
'''

with open(filename,'w') as fp:
    fp.write(script)


# In[23]:


#This is the actual REGENIE bash script
filename2='2_regenie_aous_pp.sh'

script = '''
set -o pipefail 
set -o errexit

#This imports in the actual plink genetic files
echo "${bedfile}"
echo "${bimfile}"
echo "${famnfile}"

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
'''

with open(filename2,'w') as fp:
    fp.write(script)


# In[24]:


#Upload to GCP Bucket
get_ipython().system('gsutil cp ./1_regenie_plinkprep.sh ./2_regenie_aous_pp.sh -I {my_bucket}/dsub/scripts/')


# In[25]:


#Check the files are there
get_ipython().system('gsutil ls {my_bucket}/dsub/scripts/*.sh')


# ## Dsub Submission Script
# 
# First I submit and monitor the plink script.
# 
# Note that all --input files have to be in double quotations whereas all --env environmental variables (for dsub) are NOT in quotations e.g --env chrom=${chromo}
# 
# Bash environmental variables are in "${...}" format 

# In[19]:


get_ipython().system('echo "${ARTIFACT_REGISTRY_DOCKER_REPO}" #Base location for public Docker images via Dockerhub')


# In[26]:


# !gsutil -u $GOOGLE_PROJECT ls gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/plink_bed/


# In[27]:


#This submits the job btw for each chromosome!
get_ipython().run_line_magic('%bash', '--out LINE_COUNT_JOB_ID')

# Get a shorter username to leave more characters for the job name.
DSUB_USER_NAME="$(echo "${OWNER_EMAIL}" | cut -d@ -f1)"

# For AoU RWB projects network name is "network".
AOU_NETWORK=network
AOU_SUBNETWORK=subnetwork

MACHINE_TYPE="n2-standard-4"
BASH_SCRIPT="gs://fc-secure-7953e92c-a6a6-42df-9f19-86d553a9044f/dsub/scripts/1_regenie_plinkprep.sh" #From above command

# Python is 'right side limited' wherein the last value is not included
# To run the regression across all chromosomes, set lower to 1 and upper to 23
# To run across one chromosome, set lower to the chomosome-of-interest and upper to the following

LOWER=1
UPPER=23
for ((chromo=$LOWER;chromo<$UPPER;chromo+=1))
do
    dsub \
    --provider google-cls-v2 \
    --user-project "${GOOGLE_PROJECT}" \
    --project "${GOOGLE_PROJECT}" \
    --image "us.gcr.io/broad-dsp-gcr-public/terra-jupyter-aou:2.2.14" \
    --network "${AOU_NETWORK}" \
    --subnetwork "${AOU_SUBNETWORK}" \
    --service-account "$(gcloud config get-value account)" \
    --user "${DSUB_USER_NAME}" \
    --regions us-central1 \
    --logging "${WORKSPACE_BUCKET}/dsub/logs/{job-name}/{user-id}/$(date +'%Y%m%d/%H%M%S')/{job-id}-{task-id}-{task-attempt}.log" \
    "$@" \
    --preemptible \
    --boot-disk-size 1000 \
    --machine-type ${MACHINE_TYPE} \
    --name "${JOB_NAME}" \
    --script "${BASH_SCRIPT}" \
    --env GOOGLE_PROJECT=${GOOGLE_PROJECT} \
    --input bedfile="gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/plink_bed/acaf_threshold.chr${chromo}.bed" \
    --input bimfile="gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/plink_bed/acaf_threshold.chr${chromo}.bim" \
    --input famfile="gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/plink_bed/acaf_threshold.chr${chromo}.fam" \
    --env chrom=${chromo} \
    --output-recursive OUTPUT_PATH="${OUTPUT_FILES}/${chromo}"
done


# In[28]:


# Check the status of your job submissions

get_ipython().system("dstat      --provider google-cls-v2      --project terra-vpc-sc-4e1b6fe8      --location us-central1      --jobs '*'      --users 'jon126'      --status '*'     # --full")

