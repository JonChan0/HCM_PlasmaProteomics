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
# python 1_Polyproteomic_CaseControl_Train.py --model logistic_regression --output_folder ../OUTPUT/UKB/ML/1_LR

# echo  'Training model for l1_logistic_regression'
# python 1_Polyproteomic_CaseControl_Train.py --model l1_logistic_regression --output_folder ../OUTPUT/UKB/ML/2_L1_LR

# echo 'Training model for random forest'
# python 1_Polyproteomic_CaseControl_Train.py --model random_forest --output_folder ../OUTPUT/UKB/ML/3_RF

echo 'Training model for random forest without feature selection'
python 1_Polyproteomic_CaseControl_Train.py --model random_forest_no_fs --output_folder ../OUTPUT/UKB/ML/3_RF

# echo 'Training model for xgboost'
# python 1_Polyproteomic_CaseControl_Train.py --model xgboost --output_folder ../OUTPUT/UKB/ML/4_XGB

# echo 'Training model for svm'
# python 1_Polyproteomic_CaseControl_Train.py --model svm --output_folder ../OUTPUT/UKB/ML/5_SVM
