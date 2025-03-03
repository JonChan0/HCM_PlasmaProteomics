#!/bin/bash
#SBATCH -p short
#SBATCH -A procardis.prj
#SBATCH --constraint="skl-compat"
#SBATCH -c 2
#SBATCH --cpus-per-task=2

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate python3.11_ml

# Set the number of threads for Python
export OMP_NUM_THREADS=2

model_name=xgboost #Definet the model name here e.g logistic_regression, xgboost, svm, l1_logistic_regression_nofs
echo Performing SHAP computation for trained model $model_name

python 5_Polyproteomic_SHAP.py \
    --model_pkl_file ../OUTPUT/UKB/ML/2_models/1_hcm_cc_noprs/"$model_name"_best_model.pkl \
    --model_name xgboost \
    --X_train_preprocessed_path ../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_train_preprocessed_"$model_name".csv \
    --y_train_path ../../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/y_train.csv \
    --plot_output_path ../OUTPUT/UKB/ML/3_summary_plots/feature_importance/1_hcm_cc_noprs/shap \
    --pp_names_file=../DATA/UKB/ML/2_covariates_pp/ML_pp_names.csv