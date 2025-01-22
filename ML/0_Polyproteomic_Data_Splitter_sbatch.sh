#!/bin/bash
#SBATCH -p short
#SBATCH -A procardis.prj
#SBATCH --constraint="skl-compat"
#SBATCH -c 2

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate python3.11_ml

# Set the number of threads for Python
export OMP_NUM_THREADS=2

#Scenario 1: Case control status prediction (5Y)
# python 0_Polyproteomic_Data_Splitter.py \
#     --data_path '../DATA/UKB/PROCESSED/2_5Y/ukb_ALLpanel_pp_hcm_covariates_bp_t2d_smoking_rcmmcov.tsv' \
#     --data_output_folder '../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/' \
#     --target_variable 'prevalent' 

# Scenario 2: Incident HCM prediction from the 5Y controls
python 0_Polyproteomic_Data_Splitter.py \
    --data_path '../DATA/UKB/PROCESSED/2_5Y/ukb_ALLpanel_pp_hcm_covariates_bp_t2d_smoking_rcmmcov_HCMdiagCox.tsv' \
    --data_output_folder '../OUTPUT/UKB/ML/1_data/3_hcm_incident_noprs/' \
    --target_variable 'incident' 

#N.B You have to append the cv_prs column manually to the X_train, X_test, y_train, y_test columns afterwards