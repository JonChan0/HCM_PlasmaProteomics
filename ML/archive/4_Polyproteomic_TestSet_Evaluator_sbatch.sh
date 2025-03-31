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

#Define the model name here e.g logistic_regression, xgboost, svm, l1_logistic_regression_nofs
# model_name=xgboost_nofs 
model_name=xgboost_nofs_cov_ccFDR9only
echo Performing test set evaluation $model_name

python 4_Polyproteomic_TestSet_Evaluator.py \
    --model_pkl_file ../OUTPUT/UKB/ML/2_models/1_hcm_cc_noprs/"$model_name"_best_model.pkl \
    --model_name $model_name \
    --X_test_path ../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_test.csv \
    --y_test_path ../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/y_test.csv \
    --plot_output_path ../OUTPUT/UKB/ML/3_summary_plots/test_set_inference/1_hcm_cc_noprs/ \
    --n_boostraps 1000 \
    --random_seed 42
