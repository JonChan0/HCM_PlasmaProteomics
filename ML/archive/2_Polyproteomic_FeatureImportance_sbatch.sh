#!/bin/bash
#SBATCH -p short
#SBATCH -A procardis.prj
#SBATCH --constraint="skl-compat"
#SBATCH -c 4
#SBATCH --cpus-per-task=4


module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate python3.11_ml

# Set the number of threads for Python
export OMP_NUM_THREADS=4

model_name=xgboost #Definet the model name here e.g logistic_regression, xgboost, svm, l1_logistic_regression_nofs

model_pkl_file=../OUTPUT/UKB/ML/2_models/1_hcm_cc_noprs/"$model_name"_best_model.pkl
output_folder=../OUTPUT/UKB/ML/3_summary_plots/feature_importance/1_hcm_cc_noprs/
X_data_file=../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_train_preprocessed_"$model_name".csv
y_data_file=../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/y_train.csv
pp_names_file=../DATA/UKB/ML/2_covariates_pp/ML_pp_names.csv

echo 'Evaluating feature importance for' $model_name
python 2_Polyproteomic_FeatureImportance_Evaluator.py \
    --model_name $model_name \
    --model_pkl_file $model_pkl_file \
    --output_folder $output_folder \
    --X_data_file $X_data_file \
    --y_data_file $y_data_file \
    --pp_names_file $pp_names_file
