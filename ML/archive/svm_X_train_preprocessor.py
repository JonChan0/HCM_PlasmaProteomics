import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import argparse
from sklearn.impute import KNNImputer
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sklearn.set_config(transform_output="pandas")

def load_model(model_path):
    """Load the saved model from a .pkl file."""
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)

#Customise parameters here
model_folder = '../OUTPUT/UKB/ML/2_models/1_hcm_cc_noprs/'
plot_output_folder = '../OUTPUT/UKB/ML/summary_plots/feature_importance/'
model_file = 'svm_best_model.pkl'
feature_selected='True'

#Output a preprocessed version of the original X_train dataset for SVM in particular
X_train =pd.read_csv('../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_train.csv')

model_path = os.path.join(model_folder, model_file)
model_name = os.path.splitext(model_file)[0]
model = load_model(model_path)

# Get feature names after preprocessing
preprocessor1 = model.named_steps['imputer_preprocessor']
preprocessor1

# Get feature names after preprocessing
preprocessor2 = model.named_steps['feature_preprocessor']
preprocessor2

X_train_preprocessed = preprocessor1.transform(X_train)
X_train_preprocessed = preprocessor2.transform(X_train_preprocessed)
base_folder = os.path.dirname('../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_train.csv')
preprocessed_data_path = os.path.join(base_folder, 'X_train_preprocessed_svm.csv')
pd.DataFrame(X_train_preprocessed, columns=preprocessor2.get_feature_names_out()).to_csv(preprocessed_data_path, index=False)
print('Done')