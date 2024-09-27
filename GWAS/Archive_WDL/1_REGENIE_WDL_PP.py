#!/usr/bin/env python
# coding: utf-8

# # REGENIE GWAS via WDL
# 
# This runs a GWAS of a phenotype using REGENIE via WDL workflow in GCP Terra using All of Us srWGS dataset (ACAF threshold).
# 
# Base code derived from https://workbench.researchallofus.org/workspaces/aou-rw-c241fa63/howtorunwdlsusingcromwellintheresearcherworkbenchv7/analysis/preview/02.%20Validate_VCFs_with_Cromwell_Server_Mode.ipynb
# 
# This uses an existing Cromwell Cloud environment instance in All of Us.
# 

# ## Setup

# In[1]:


#use the datetime function to measure how long the workflow takes
#set up workspace bucket variable to be used later
from datetime import datetime
import os
start = datetime.now()
bucket = os.getenv("WORKSPACE_BUCKET")
bucket


# In[2]:


import os
import subprocess
import json
import requests
from pathlib import Path

def check_for_app(env):
    list_apps_url = f'{env["leonardo_url"]}/api/google/v1/apps/{env["google_project"]}'
    r = requests.get(
        list_apps_url,
        params={
          'includeDeleted': 'false'
        },
        headers = {
            'Authorization': f'Bearer {env["token"]}'
        }
    )
    r.raise_for_status()

    for potential_app in r.json():
        if potential_app['appType'] == 'CROMWELL' and (
                str(potential_app['auditInfo']['creator']) == env['owner_email']
                or str(potential_app['auditInfo']['creator']) == env['user_email']
        ) :
            potential_app_name = potential_app['appName']
            potential_app_status = potential_app['status']

            # We found a CROMWELL app in the correct google project and owned by the user. Now just check the workspace:
            _, workspace_namespace,  proxy_url = get_app_details(env, potential_app_name)
            if workspace_namespace == env['workspace_namespace']:
                return potential_app_name, potential_app_status, proxy_url['cromwell-service']

    return None, None, None

def get_app_details(env, app_name):
    get_app_url = f'{env["leonardo_url"]}/api/google/v1/apps/{env["google_project"]}/{app_name}'
    print('start')
    r = requests.get(
        get_app_url,
        params={
            'includeDeleted': 'true',
            'role': 'creator'
        },
        headers={
            'Authorization': f'Bearer {env["token"]}'
        }
    )
    if r.status_code == 404:
        return 'DELETED', None, None, None
    else:
        r.raise_for_status()
    result_json = r.json()
    custom_environment_variables = result_json['customEnvironmentVariables']
    return result_json['status'], custom_environment_variables['WORKSPACE_NAMESPACE'], result_json.get('proxyUrls')

# Checks that cromshell is installed. Otherwise raises an error.
def validate_cromshell():
    if validate_cromshell_alias():
        print("Found cromshell, please use cromshell")
    elif validate_cromshell_alpha():
        print("Found cromshell-alpha, please use cromshell-alpha")
    else:
        raise Exception("Cromshell is not installed.")

# Checks that cromshell is installed. Otherwise raises an error.
def validate_cromshell_alpha():
    print('Scanning for cromshell 2 alpha...')
    try:
        subprocess.run(['cromshell-alpha', 'version'], capture_output=True, check=True, encoding='utf-8')
    except FileNotFoundError:
        return False
    return True
# Checks that cromshell is installed. Otherwise raises an error.
def validate_cromshell_alias():
    print('Scanning for cromshell 2')
    try:
        subprocess.run(['cromshell', 'version'], capture_output=True, check=True, encoding='utf-8')
    except FileNotFoundError:
        return False
    return True

def configure_cromwell(env, proxy_url):
     print('Updating cromwell config')
     file = f'{str(Path.home())}/.cromshell/cromshell_config.json'
     configuration = {
        'cromwell_server': proxy_url.split("swagger/", 1)[0] if proxy_url else "invalid url",
        'requests_timeout': 5,
        'gcloud_token_email': env['user_email'],
        'referer_header_url': env['leonardo_url']
     }
     with open(file, 'w') as filetowrite:
        filetowrite.write(json.dumps(configuration, indent=2))

def find_app_status(env):
    print(f'Checking status for CROMWELL app')
    app_name, app_status, proxy_url = check_for_app(env)

    configure_cromwell(env, proxy_url)

    if app_name is None:
        print(f'CROMWELL app does not exist. Please create cromwell server from workbench')
    else:
        print(f'app_name={app_name}; app_status={app_status}')
        print(f'Existing CROMWELL app found (app_name={app_name}; app_status={app_status}).')
        exit(1)

def main():
    # Iteration 1: these ENV reads will throw errors if not set.
    env = {
        'workspace_namespace': os.environ['WORKSPACE_NAMESPACE'],
        'workspace_bucket': os.environ['WORKSPACE_BUCKET'],
        'user_email': os.environ.get('PET_SA_EMAIL', default = os.environ['OWNER_EMAIL']),
        'owner_email': os.environ['OWNER_EMAIL'],
        'google_project': os.environ['GOOGLE_PROJECT'],
        'leonardo_url': os.environ['LEONARDO_BASE_URL']
    }

    # Before going any further, check that cromshell2 is installed:
    validate_cromshell()

    # Fetch the token:
    token_fetch_command = subprocess.run(['gcloud', 'auth', 'print-access-token', env['user_email']], capture_output=True, check=True, encoding='utf-8')
    env['token'] = str.strip(token_fetch_command.stdout)

    find_app_status(env)


if __name__ == '__main__':
    main()
 


# In[3]:


#Get the newest version of cromshell
get_ipython().system('python -V')

# run these to update cromshell
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install --upgrade aiohttp')
get_ipython().system('pip install git+https://github.com/broadinstitute/cromshell')


# In[4]:


get_ipython().system('cromshell version')


# In[5]:


get_ipython().system('cromshell validate --help')


# ## Load WDL and JSON Files 
# 
# This step uses your WDL file and json file to define the inputs into WDL workflow.

# In[6]:


# Import your WDL script

wdl_filename = "regenie_aous_pp.wdl"

WDL_content = """
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

"""
fp = open(wdl_filename, 'w')
fp.write(WDL_content)
fp.close()
#print(WDL_content)


# In[15]:


# Scrape the filenames from the controlled tier plink_bucket to define the bed filenames
import re

get_ipython().system("gsutil -u $GOOGLE_PROJECT ls 'gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/plink_bed/' > plink_filepaths.txt")

with open('plink_filepaths.txt', 'r') as plink_filepaths :
    plink_filepaths = plink_filepaths.read()
    plink_bednames = re.findall('(.+[^XY]\\.bed)',plink_filepaths)
    plink_bimnames = re.findall('(.+[^XY]\\.bim)',plink_filepaths)
    plink_famnames = re.findall('(.+[^XY]\\.fam)',plink_filepaths)
    
print(plink_bednames,plink_bimnames, plink_famnames)


# In[48]:


#Provide your json file to define your inputs
# controlled_tier_bucket_plink = 'gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/plink_bed/'
pheno_of_choice = 'ntprobnp'

json_filename = 'regenie_aous_'+pheno_of_choice+'.json'

#N.B Json syntax requires double quotes and key:name pairs

#Derive each line of the json inputs
json_bed = '\"regenie_aous.bed_files\": [' + ', '.join([f'"{item}"' for item in plink_bednames]) + "]"
json_bim = '\"regenie_aous.bim_files\": [' + ', '.join([f'"{item}"' for item in plink_bimnames]) + "]"
json_fam = '\"regenie_aous.fam_files\": [' + ', '.join([f'"{item}"' for item in plink_famnames]) + "]"
json_pheno = '\"regenie_aous.pheno_file\": ' +'\"' + bucket+ '/PP/DATA/pp_regenie_pheno.tsv\"'
json_covar = '\"regenie_aous.covar_file\": ' +'\"' + bucket+ '/PP/DATA/pp_regenie_covar.tsv\"'
json_choice = '\"regenie_aous.pheno_of_choice\": ' + '\"' + pheno_of_choice +'\"'

json_content = f'''
{{
    {json_bed},
    {json_bim},
    {json_fam},
    {json_pheno},
    {json_covar},
    {json_choice}
}}
'''

fp = open(json_filename, 'w')
fp.write(json_content)
fp.close()


# ## WDL Execution via Cromshell

# In[49]:


##Submit the regenie_aous_pp.wdl to Cromwell
get_ipython().system('cromshell submit regenie_aous_pp.wdl regenie_aous_ntprobnp.json')


# In[50]:


## Check submission ID
#Get the most recent submission id as $submission_id
with open('/home/jupyter/.cromshell/all.workflow.database.tsv') as f:
    for line in f:
        pass
    most_recent_submission = line.strip().split('\t')
submission_id = most_recent_submission[2]
print(submission_id)


# In[51]:


#Check status of submission to Cromwell
get_ipython().system('cromshell status $submission_id')


# In[53]:


#Upload this script 
# get the bucket name
my_bucket = os.getenv('WORKSPACE_BUCKET')

# copy csv file to the bucket
args = ["gsutil", "cp", f"./1_REGENIE_WDL_PP.ipynb", f"{my_bucket}/PP/GWAS/"]
output = subprocess.run(args, capture_output=True)

# print output from gsutil
output.stderr


# In[ ]:


#This function checks the status of the Cromwell job submission

import time
def check_job_status(submission_id):
    while True:
        try:
            # Run the command and capture the output
            result = subprocess.run(
                f"cromshell status {submission_id}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            # Extract the status from the output
            output = result.stdout
            status_start = output.find("status\":\"") + len("status\":\"")
            status_end = output.find("\"", status_start)
            status = output[status_start:status_end]

            print(f"Job status: {status}")

            if status == "Succeeded":
                break  # Job has succeeded, exit the loop

        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
        
        time.sleep(120)  # Wait for 2 minutes before checking again

# Call the function to check the job status
check_job_status(submission_id)

