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

model_output_folder='../OUTPUT/UKB/ML/2_models/1_hcm_cc_noprs/'
features_to_bypass='../DATA/UKB/ML/2_covariates_pp/ML_quantitative_covariates.csv'  #This refers to the quantitative features you should NOT pass to feature selection step
features_to_select='../DATA/UKB/ML/2_covariates_pp/ML_pp_names.csv'  #This refers to the quantitative features you SHOULD pass to feature selection step
target_variable='prevalent'
X_train_data_path='../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_train.csv'
y_train_data_path='../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/y_train.csv'

# echo 'Training model for logistic regression'
# python 1_Polyproteomic_ML_Train.py --model_name logistic_regression --plot_output_folder "$model_output_folder"1_LR \
#     --feature_selection True \
#     --model_output_folder $model_output_folder \
#     --features_to_bypass_fs $features_to_bypass --features_to_select_fs $features_to_select  \
#     --target_variable $target_variable --X_train_data $X_train_data_path  --y_train_data $y_train_data_path

# echo  'Training model for l1_logistic_regression'
# python 1_Polyproteomic_ML_Train.py --model_name l1_logistic_regression --plot_output_folder "$model_output_folder"2_L1_LR \
#     --feature_selection False \
#     --model_output_folder $model_output_folder\
#     --features_to_bypass_fs $features_to_bypass --features_to_select_fs $features_to_select  \
#     --target_variable $target_variable --X_train_data $X_train_data_path --y_train_data $y_train_data_path

# echo 'Training model for random forest'
# python 1_Polyproteomic_ML_Train.py --model_name random_forest --plot_output_folder "$model_output_folder"3_RF \
#     --feature_selection True \
#     --model_output_folder $model_output_folder\
#     --features_to_bypass_fs $features_to_bypass --features_to_select_fs $features_to_select  \
#     --target_variable $target_variable --X_train_data $X_train_data_path --y_train_data $y_train_data_path

# echo 'Training model for xgboost'
# python 1_Polyproteomic_ML_Train.py --model_name xgboost --plot_output_folder "$model_output_folder"4_XGB/ \
#     --feature_selection True \
#     --model_output_folder $model_output_folder\
#     --features_to_bypass_fs $features_to_bypass --features_to_select_fs $features_to_select  \
#     --target_variable $target_variable --X_train_data $X_train_data_path --y_train_data $y_train_data_path

# echo 'Training model for xgboost with ONLY covariates + NTproBNP'
# model_output_folder='../OUTPUT/UKB/ML/2_models/1_hcm_cc_noprs/covariates_NTproBNPonly/'
# features_to_select='../DATA/UKB/ML/2_covariates_pp/NTproBNP_names.csv'
# python 1_Polyproteomic_ML_Train.py --model_name xgboost --plot_output_folder "$model_output_folder"4_XGB/ \
#     --feature_selection False \
#     --model_output_folder $model_output_folder\
#     --features_to_bypass_fs $features_to_bypass --features_to_select_fs $features_to_select  \
#     --target_variable $target_variable --X_train_data $X_train_data_path --y_train_data $y_train_data_path

# echo 'Training model for xgboost with ONLY covariates + 9 FDR-passing plasma proteins from case-control analysis'
# model_output_folder='../OUTPUT/UKB/ML/2_models/1_hcm_cc_noprs/covariates_cc_top9/'
# features_to_select='../DATA/UKB/ML/2_covariates_pp/cc_top9_names.csv'
# python 1_Polyproteomic_ML_Train.py --model_name xgboost --plot_output_folder "$model_output_folder"4_XGB/ \
#     --feature_selection False \
#     --model_output_folder $model_output_folder\
#     --features_to_bypass_fs $features_to_bypass --features_to_select_fs $features_to_select  \
#     --target_variable $target_variable --X_train_data $X_train_data_path --y_train_data $y_train_data_path

# echo 'Training model for svm'
# python 1_Polyproteomic_ML_Train.py --model_name svm --plot_output_folder "$model_output_folder"5_SVM \
#     --feature_selection True \
#     --model_output_folder $model_output_folder\
#     --features_to_bypass_fs $features_to_bypass --features_to_select_fs $features_to_select  \
#     --target_variable $target_variable --X_train_data $X_train_data_path --y_train_data $y_train_data_path

# echo 'Training model for PLS as feature selection step + LDA for binary classification = PLS-LDA'
# python 1_Polyproteomic_ML_Train.py --model_name spls_lda --plot_output_folder "$model_output_folder"6_PLS_LDA \
#     --feature_selection False \
#     --model_output_folder $model_output_folder\
#     --features_to_bypass_fs $features_to_bypass --features_to_select_fs $features_to_select  \
#     --target_variable $target_variable --X_train_data $X_train_data_path --y_train_data $y_train_data_path

# echo 'Training model for elastic_net logistic regression'
# python 1_Polyproteomic_ML_Train.py --model_name elastic_net_logistic_regression --plot_output_folder "$model_output_folder"7_EL_LR \
#     --feature_selection False \
#     --model_output_folder $model_output_folder\
#     --features_to_bypass_fs $features_to_bypass --features_to_select_fs $features_to_select  \
#     --target_variable $target_variable --X_train_data $X_train_data_path --y_train_data $y_train_data_path