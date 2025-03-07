'''
Script to evaluate the SHAP values of the trained model on the entire dataset and output the SHAP summary plot.
Author: Jonathan Chan
Date: 2025-02-28
'''

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import argparse
import sklearn
from adjustText import adjust_text
from sklearn.feature_selection import f_classif

sklearn.set_config(transform_output="pandas")

def load_model(model_path):
    """Load the saved model from a .pkl file."""
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)

# Filter for specific features by name
def filter_shap_values(shap_values, feature_names, features_to_keep):
    # Get indices of features to keep
    indices = [feature_names.index(feature) for feature in features_to_keep]
    
    # Create a filtered version of the SHAP values
    filtered_values = shap_values.values[:, indices]
    
    # Create a new SHAP values object
    filtered_shap_values = shap.Explanation(
        values=filtered_values,
        base_values=shap_values.base_values,
        data=shap_values.data[:, indices],
        feature_names=[feature_names[i] for i in indices]
    )
    
    return filtered_shap_values

def plot_shap_plots(shap_values, model_name, n_features, output_folder, suffix=''):
    """Plot SHAP plots for the given model."""
    print(f"Plotting SHAP plots for {model_name}")

    #Output shap barplot displaying all
    shap.plots.bar(shap_values, show=False,max_display=n_features)
    plt.title(f'Absolute SHAP Values for {n_features} features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{model_name}_shap_barplot{suffix}.png"))
    plt.close()

    # Plot SHAP beeswarm plot
    shap.plots.beeswarm(shap_values, max_display=n_features, show=False)
    plt.title(f'SHAP beeswarm plot for {n_features} features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{model_name}_shap_beeswarm{suffix}.png"))
    plt.close()

def dependence_shap_plotter(shap_values, model_name, output_folder, top_n=5):
    
    # Assuming shap_values is your SHAP explanation object
    # Get feature importance
    feature_importance = np.abs(shap_values.values).mean(0)
    top_indices = np.argsort(-feature_importance)[:top_n]  # Get indices of top 5 features
    
    # Create a figure with 5 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()  # Flatten for easier indexing
    
    # Plot each feature dependence plot in its own subplot
    for i, feature_idx in enumerate(top_indices):
        if i < len(axes):
            ax = axes[i]
            shap.plots.scatter(
                shap_values[:, feature_idx],
                #color=shap_values,
                show=False,
                ax=ax
            )
            ax.set_title(f'Feature: {shap_values.feature_names[feature_idx]}', fontsize=12)
            ax.axhline(y=0,c='black',linestyle='--')
    
    # Hide the unused subplot if any
    if len(top_indices) < len(axes):
        axes[-1].set_visible(False)
    
    # Add an overall title to the figure
    fig.suptitle(f'SHAP Dependence Plots for Top {top_n} Features', fontsize=16, y=0.98)
    plt.savefig(os.path.join(output_folder, f'{model_name}_top{top_n}_dependence_plots.png'), dpi=300, bbox_inches='tight')
    # plt.show()

def shap_to_df(shap_values):
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Create DataFrame with feature names and their importance
    feature_importance = pd.DataFrame({
        'Feature': shap_values.feature_names,
        'mean_abs_shap_value': mean_abs_shap
    })
    
    # Sort by importance (highest to lowest)
    feature_importance = feature_importance.sort_values('mean_abs_shap_value', ascending=False)
    
    return feature_importance

def plot_shap_vs_Fratio(mean_shap_df, fratio_df, model_name, output_folder, top_n_label=5):
    """Plot mean absolute SHAP-values vs F-ratio for the given model. It also labels the top 5 features for both SHAP and F ratio"""
    print(f"Plotting mean absolute SHAP-values vs F-ratio for {model_name}")

    # Merge the feature importances and F-ratio dataframes
    merged_df = mean_shap_df.merge(fratio_df, on='Feature', how='inner')

    #Add the label column with True if in top 5 features for either SHAP or Fratio
    top5_fratio = fratio_df.sort_values('F-ratio',ascending=False).iloc[:top_n_label+1,:]['Feature'].values
    top5_shap = mean_shap_df.sort_values('mean_abs_shap_value',ascending=False).iloc[:top_n_label+1,:]['Feature'].values

    merged_df['label'] = merged_df['Feature'].apply(
        lambda x: x if (x in top5_fratio or x in top5_shap) else '')
    
    plt.figure(figsize=(12, 9))
    sns.scatterplot(x='mean_abs_shap_value', y='F-ratio', data=merged_df)

    # Collect texts
    texts = []
    for i, row in merged_df.iterrows():
        texts.append(plt.text(row['mean_abs_shap_value'], row['F-ratio'], row['label']))
    
    # Adjust label positions to minimize overlaps
    adjust_text(texts)
    
    plt.title(f'Mean Absolute SHAP value vs F-ratio - {model_name}')
    plt.xlabel('Mean Absolute SHAP value')
    plt.ylabel('F-ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{model_name}_shap_vs_fratio.png'))
    # plt.show()
    plt.close()
    print(f"Feature importance vs F-ratio plot saved for {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a saved model on a test set and output performance plots.')
    parser.add_argument('--model_pkl_file', type=str, required=True, help='Path to the saved model .pkl file')
    parser.add_argument('--X_train_preprocessed_path', type=str, required=True, help='Path to the X_train_preprocessed file')
    parser.add_argument('--y_train_path',type=str, required=True, help='Path to the y_train file')
    parser.add_argument('--plot_output_path', type=str, required=True, help='Path to save the output plots')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model for plot titles and filenames')
    parser.add_argument('--pp_names_file', type=str, required=True, help="CSV file containing the plasma protein names")
    args = parser.parse_args()

    # Create the output folder if it does not exist
    if not os.path.exists(args.plot_output_path):
        os.makedirs(args.plot_output_path)
        print(f"Created output folder: {args.plot_output_path}")

    # Load the data
    print(f"Loading data from {args.X_train_preprocessed_path}")
    X = pd.read_csv(args.X_train_preprocessed_path)
    print(f"Preprocessed training data loaded with shape: {X.shape}")

    pp_names = pd.read_csv(args.pp_names_file).iloc[:,0].values

    model = load_model(args.model_pkl_file)
    explainer = shap.Explainer(model.named_steps['classifier'])
    shap_values = explainer(X)

    # Plot SHAP summary plot for all features in the model
    if len(X.columns) <= 50: #Only plot all the features if the number of total features is <= 50
        plot_shap_plots(shap_values, args.model_name, len(X.columns), args.plot_output_path)
    plot_shap_plots(shap_values, args.model_name, 10, args.plot_output_path, '_top10') #Also plot only showing top 10 features

    #Plot SHAP plots for only filtered features i.e plasma proteins
    pp_names = pp_names[np.isin(pp_names, X.columns.values)]
    print(f'The filtered features include {pp_names}')
    shap_values_filtered = filter_shap_values(shap_values, X.columns.values.tolist(), pp_names) #SHAP values for only the plasma proteins

    if len(pp_names) <= 50: #Only plot all the features if the number of total features is <= 50
        plot_shap_plots(shap_values_filtered, args.model_name, len(pp_names), args.plot_output_path, '_ppfiltered')

    plot_shap_plots(shap_values_filtered, args.model_name, 10, args.plot_output_path, '_ppfiltered_top10') #Also plot only showing top 10 plasma proteins
    plot_shap_plots(shap_values_filtered, args.model_name, 30, args.plot_output_path, '_ppfiltered_top30') #Also plot only showing top 30 plasma proteins

    #Plot the SHAP dependence plots for the top_n_features
    dependence_shap_plotter(shap_values_filtered, args.model_name, args.plot_output_path, top_n=5)

    #Plot the SHAP values against the F-ratio
    #Also plot a F-value plot vs. mean absolute SHAP value plot

    #Compute the F-ratio for the plasma proteins in the X_train data file given that X is a dataframe
    fvalues = f_classif(X.loc[:,pp_names], pd.read_csv(args.y_train_path).values.ravel())[0]
    fratio_df = pd.DataFrame({'Feature': pp_names, 'F-ratio': fvalues})
    mean_shap_filtered_df = shap_to_df(shap_values_filtered)
    plot_shap_vs_Fratio(mean_shap_filtered_df, fratio_df, args.model_name, args.plot_output_path)