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

def plot_feature_importance(model, feature_names, model_name, output_folder, features_filtered=None, n_features_to_plot=30):
    """Plot feature importance for the given model."""
    print(f"Plotting feature importance for {model_name}")

    num_features = len(feature_names)

    if num_features < n_features_to_plot * 2:  # If the actual number of features after selection is close but > n_features_to_plot
        n_features_to_plot = num_features + 1

    if hasattr(model, 'feature_importances_') or (hasattr(model, 'named_steps') and 'classifier' in model.named_steps and hasattr(model.named_steps['classifier'], 'feature_importances_')):
        plot_label = 'Importances'
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = model.named_steps['classifier'].feature_importances_
    elif hasattr(model, 'coef_') or (hasattr(model, 'named_steps') and 'classifier' in model.named_steps and hasattr(model.named_steps['classifier'], 'coef_')):
        plot_label = 'Coefficients'
        
        if hasattr(model, 'coef_'):
            importances = model.coef_[0]
        else:
            importances = model.named_steps['classifier'].coef_[0]
            
    if features_filtered is not None:
        # Filter the features if features_filtered is defined
        mask = np.isin(feature_names, features_filtered)
        feature_names = feature_names[mask]
        importances = importances[mask]
        plot_suffix = '_featurefiltered'
    else:
        plot_suffix=''

    #Sort the features by importance + filter to the n_features_to_plot
    indices = np.argsort(importances)[::-1]
    sorted_features = feature_names[indices][:n_features_to_plot]
    sorted_importances = importances[indices][:n_features_to_plot]

    plt.figure(figsize=(12, 9))
    sns.barplot(x=sorted_importances, y=sorted_features)
    plt.title(f'Feature {plot_label} - {model_name}')
    plt.xlabel(f'{plot_label}')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{model_name}_feature_importances{plot_suffix}.png'))
    plt.close()
    print(f"Feature coefficients plot saved for {model_name}")

    #Output the feature importances as a csv file with the respective feature names as well
    feature_importances_df = pd.DataFrame({'Feature': sorted_features, 'Importance': sorted_importances})
    feature_importances_df.to_csv(os.path.join(output_folder, f'{model_name}_feature_importances{plot_suffix}.csv'), index=False)
    print(f"Feature importances saved for {model_name}")

    return feature_importances_df

def plot_Fratio(fratio_df, model_name,output_folder, features_filtered=None):
    """Plot the F-ratio for the given model."""
    print(f"Plotting F-ratio for {model_name}")

    fratio_df = fratio_df.sort_values('F-ratio', ascending=False)

    if features_filtered is not None:
        mask =np.isin(features_filtered,fratio_df['Feature'])
        features_of_interest=features_filtered[mask]
        fratio_df = fratio_df[fratio_df['Feature'].isin(features_of_interest)]

    plt.figure(figsize=(12, 9))
    sns.barplot(x='F-ratio', y='Feature', data=fratio_df)
    plt.title(f'F-ratio - {model_name}')
    plt.xlabel('F-ratio')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{model_name}_fratio.png'))
    plt.close()
    print(f"F-ratio plot saved for {model_name}")

def plot_feature_importance_vs_Fratio(feature_importances_df, fratio_df, model_name, output_folder):
    """Plot feature importance vs F-ratio for the given model."""
    print(f"Plotting feature importance vs F-ratio for {model_name}")

    # Merge the feature importances and F-ratio dataframes
    merged_df = feature_importances_df.merge(fratio_df, on='Feature', how='inner')

    plt.figure(figsize=(12, 9))
    sns.scatterplot(x='Importance', y='F-ratio', data=merged_df)
    plt.title(f'Feature Importance vs F-ratio - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('F-ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{model_name}_feature_importance_vs_fratio.png'))
    plt.close()
    print(f"Feature importance vs F-ratio plot saved for {model_name}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate feature importance for saved models.')
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model")
    parser.add_argument('--model_pkl_file', type=str, required=True, help="Folder path containing the saved models")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder path to save the feature importance plots")
    parser.add_argument('--X_data_file', type=str, required=True, help="CSV file containing the preprocessed data used for training")
    parser.add_argument('--y_data_file', type=str, required=True, help="CSV file containing the target labels")
    parser.add_argument('--pp_names_file', type=str, required=True, help="CSV file containing the plasma protein names")
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
    
    #This plots the feature importance for all the features i.e both clinical covariates and plasma proteins
    plot_feature_importance(model,feature_names, args.model_name, args.output_folder)

    #Load the plasma protein feature names
    pp_names = pd.read_csv(args.pp_names_file)['name'].values

    #Plot feature importances for only the plasma proteins
    feature_importances_df = plot_feature_importance(model, feature_names, args.model_name, args.output_folder, features_filtered=pp_names)

    #Compute the F-ratio for the plasma proteins in the X_train data file given that X is a dataframe
    fvalues = sklearn.feature_selection.f_classif(X.loc[:,pp_names], pd.read_csv(args.y_data_file).values.ravel())[0]
    fratio_df = pd.DataFrame({'Feature': pp_names, 'F-ratio': fvalues})

    plot_Fratio(fratio_df, args.model_name, args.output_folder, features_filtered=pp_names)
    plot_feature_importance_vs_Fratio(feature_importances_df, fratio_df, args.model_name, args.output_folder)

    print(f"Completed feature importance evaluation for {args.model_name}")