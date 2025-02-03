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

echo 'Performing inference on test set and evaluating test set performance'
python 3_Polyproteomic_TestSet_Evaluator.py \
    --model_name xgboost \
    --model_path ../OUTPUT/UKB/ML/2_models/1_hcm_cc_noprs/xgboost_best_model.pkl \
    --plot_output_path ../OUTPUT/UKB/ML/3_summary_plots/test_set_inference \
    --X_test_path ../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_test.csv \
    --y_test_path ../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/y_test.csv \
    --n_bootstraps 1000 \
    --random_seed 42