#!/bin/bash
#SBATCH -p long
#SBATCH -A procardis.prj
#SBATCH --constraint="skl-compat"
#SBATCH -c 8

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate python3.11_ml

# Set the number of threads for Python
export OMP_NUM_THREADS=8

#These are training different model classes with only clinical risk factors + plasma proteomics

model_output_folder='../OUTPUT/UKB/ML/2_models/5_hcm_allcases_noprs/'
features_to_bypass='../DATA/UKB/ML/2_covariates_pp/ML_quantitative_covariates.csv'  #This refers to the quantitative features you should NOT pass to feature selection step
features_to_select='../DATA/UKB/ML/2_covariates_pp/ML_pp_names.csv'  #This refers to the quantitative features you SHOULD pass to feature selection step
target_variable='allcases'
X_train_data_path='../OUTPUT/UKB/ML/1_data/5_hcm_allcases_noprs/X_train.csv'
y_train_data_path='../OUTPUT/UKB/ML/1_data/5_hcm_allcases_noprs/y_train.csv'

echo 'Training model for xgboost'
python 1_Polyproteomic_ML_Train.py --model_name xgboost --plot_output_folder ../OUTPUT/UKB/ML/2_models/5_hcm_allcases_noprs/4_XGB/ \
    --feature_selection True \
    --model_output_folder $model_output_folder\
    --features_to_bypass_fs $features_to_bypass --features_to_select_fs $features_to_select  \
    --target_variable $target_variable --X_train_data $X_train_data_path --y_train_data $y_train_data_path

