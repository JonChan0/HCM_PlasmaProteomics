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

echo 'Evaluating feature importance'
python 2_Polyproteomic_FeatureImportance_Evaluator.py \
    --model_name xgboost \
    --model_pkl_file ../OUTPUT/UKB/ML/2_models/1_hcm_cc_noprs/xgboost_best_model.pkl \
    --output_folder ../OUTPUT/UKB/ML/3_summary_plots/feature_importance/1_hcm_cc_noprs/ \
    --X_data_file ../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_train_preprocessed_xgboost.csv \
    --pp_names_file ../DATA/UKB/ML/2_covariates_pp/ML_pp_names.csv

python 2_Polyproteomic_FeatureImportance_Evaluator.py \
    --model_name xgboost \
    --model_pkl_file ../OUTPUT/UKB/ML/2_models/5_hcm_allcases_noprs/xgboost_best_model.pkl \
    --output_folder ../OUTPUT/UKB/ML/3_summary_plots/feature_importance/5_hcm_allcases_noprs/ \
    --X_data_file ../OUTPUT/UKB/ML/1_data/5_hcm_allcases_noprs/X_train_preprocessed_xgboost.csv \
    --pp_names_file ../DATA/UKB/ML/2_covariates_pp/ML_pp_names.csv
