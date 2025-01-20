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
export OMP_NUM_THREADS=8

echo 'Evaluating feature importance'
python 2_Polyproteomic_FeatureImportance_Evaluator.py --model_folder ../OUTPUT/UKB/ML/models/ --output_folder ../OUTPUT/UKB/ML/summary_plots/feature_importance/ --data_file ../OUTPUT/UKB/ML/1_LR/X_train.csv
