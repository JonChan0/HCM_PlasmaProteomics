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

# echo 'Training model for logistic regression'
# python 1_Polyproteomic_CaseControl_Train.py --model logistic_regression --plot_output_folder ../OUTPUT/UKB/ML/1_LR \
#     --feature_selection True \
#     --model_output_folder ../OUTPUT/UKB/ML/models/ \
#     --features_to_bypass_fs ../DATA/UKB/ML/ML_quantitative_covariates.csv --features_to_select_fs ../DATA/UKB/ML/ML_pp_names.csv

# echo  'Training model for l1_logistic_regression'
# python 1_Polyproteomic_CaseControl_Train.py --model l1_logistic_regression --plot_output_folder ../OUTPUT/UKB/ML/2_L1_LR \
#     --feature_selection False \
#     --model_output_folder ../OUTPUT/UKB/ML/models/ \
#     --features_to_bypass_fs ../DATA/UKB/ML/ML_quantitative_covariates.csv --features_to_select_fs ../DATA/UKB/ML/ML_pp_names.csv

# echo 'Training model for random forest'
# python 1_Polyproteomic_CaseControl_Train.py --model random_forest --plot_output_folder ../OUTPUT/UKB/ML/3_RF \
#     --feature_selection True \
#     --model_output_folder ../OUTPUT/UKB/ML/models/ \
#     --features_to_bypass_fs ../DATA/UKB/ML/ML_quantitative_covariates.csv --features_to_select_fs ../DATA/UKB/ML/ML_pp_names.csv

# echo 'Training model for xgboost'
# python 1_Polyproteomic_CaseControl_Train.py --model xgboost --plot_output_folder ../OUTPUT/UKB/ML/ \
#     --feature_selection True \
#     --model_output_folder ../OUTPUT/UKB/ML/ \
#     --features_to_bypass_fs ../DATA/UKB/ML/ML_quantitative_covariates.csv --features_to_select_fs ../DATA/UKB/ML/ML_pp_names.csv

# echo 'Training model for svm'
# python 1_Polyproteomic_CaseControl_Train.py --model svm --plot_output_folder ../OUTPUT/UKB/ML/5_SVM \
#     --feature_selection True \
#     --model_output_folder ../OUTPUT/UKB/ML/models/ \
#     --features_to_bypass_fs ../DATA/UKB/ML/ML_quantitative_covariates.csv --features_to_select_fs ../DATA/UKB/ML/ML_pp_names.csv

echo 'Training model for PLS as feature selection step + LDA for binary classification = PLS-LDA'
python 1_Polyproteomic_CaseControl_Train.py --model spls_lda --plot_output_folder ../OUTPUT/UKB/ML/6_PLS_LDA \
    --feature_selection False \
    --model_output_folder ../OUTPUT/UKB/ML/models/ \
    --features_to_bypass_fs ../DATA/UKB/ML/ML_quantitative_covariates.csv --features_to_select_fs ../DATA/UKB/ML/ML_pp_names.csv