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

def plot_feature_importance(model, X, feature_names, model_name, output_folder, n_features_to_plot=30):
    """Plot feature importance for the given model."""
    print(f"Plotting feature importance for {model_name}")

    num_features = len(feature_names)

    if(num_features < n_features_to_plot * 2): #If if the actual number of features after selection is close but > n_features_to_plot
        n_features_to_plot = num_features + 1
    
    if hasattr(model, 'feature_importances_') or (hasattr(model, 'named_steps') and 'classifier' in model.named_steps and hasattr(model.named_steps['classifier'], 'feature_importances_')):
        # Linear models (Logistic Regression, SVM)
        if hasattr(model, 'coef_'):
            importances = model.feature_importances_
        else:
            importances = model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = feature_names[indices][:n_features_to_plot]
        sorted_importances = importances[indices][:n_features_to_plot]
        
        plt.figure(figsize=(12, 9))
        sns.barplot(x=sorted_importances, y=sorted_features)
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{model_name}_feature_importances.png'))
        plt.close()
        print(f"Feature importance plot saved for {model_name}")

    elif hasattr(model, 'coef_') or (hasattr(model, 'named_steps') and 'classifier' in model.named_steps and hasattr(model.named_steps['classifier'], 'coef_')):
        # Linear models (Logistic Regression)
        if hasattr(model, 'coef_'):
            importances = model.coef_[0]
        else:
            importances = model.named_steps['classifier'].coef_[0]
        indices = np.argsort(np.abs(importances))[::-1]
        sorted_features = feature_names[indices][:n_features_to_plot]
        sorted_importances = importances[indices][:n_features_to_plot]
        
        plt.figure(figsize=(12, 9))
        sns.barplot(x=sorted_importances, y=sorted_features)
        plt.title(f'Feature Coefficients - {model_name}')
        plt.xlabel('Coefficient')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{model_name}_feature_importances.png'))
        plt.close()
        print(f"Feature coefficients plot saved for {model_name}")
    else:
        # Use SHAP values for other models
        print(f"Calculating SHAP values for {model_name}")
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        
        if 'eid' in X.columns:
            X=X.drop(columns='eid')

        background = shap.sample(X, 100, random_state=42)  # Use a sample of the data as the background
        explainer = shap.KernelExplainer(classifier.predict, background)
        shap_values = explainer.shap_values(X)
        # explainer = shap.Explainer(classifier.predict, X)
        # shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=n_features_to_plot, show=False)
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{model_name}_feature_importances.png'))
        plt.close()
        print(f"SHAP feature importance plot saved for {model_name}")

        # Save SHAP values
        shap_values_path = os.path.join(output_folder, f'{model_name}_shap_values.npy')
        np.save(shap_values_path, shap_values)
        print(f"SHAP values saved for {model_name} at {shap_values_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate feature importance for saved models.')
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model")
    parser.add_argument('--model_pkl_file', type=str, required=True, help="Folder path containing the saved models")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder path to save the feature importance plots")
    parser.add_argument('--X_data_file', type=str, required=True, help="CSV file containing the preprocessed data used for training")
    args = parser.parse_args()

    # Create the output folder if it does not exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output folder: {args.output_folder}")

    # Load the data
    print(f"Loading data from {args.X_data_file}")
    X = pd.read_csv(args.X_data_file)
    print(f"Preprocessed training data loaded with shape: {X.shape}")

    model = load_model(args.model_pkl_file)

    # Get feature names after preprocessing
    preprocessor1 = model.named_steps['imputer_preprocessor']
    # Get feature names after preprocessing
    preprocessor2 = model.named_steps['feature_preprocessor']

    feature_names = preprocessor2.get_feature_names_out()
    
    plot_feature_importance(model, X, feature_names, args.model_name, args.output_folder)
    print(f"Completed feature importance evaluation for {args.model_name}")