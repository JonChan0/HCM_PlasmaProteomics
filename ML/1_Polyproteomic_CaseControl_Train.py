import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score
import os
import wandb
from sklearn.pipeline import Pipeline
import time

# Function to define the model and parameter grid based on user input
def get_model_and_params(model_type):
    if model_type in ['logistic_regression', 'logistic_regression_no_fs']:
        model = LogisticRegression()
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__class_weight': ['balanced']
        }
    elif model_type in ['random_forest', 'random_forest_no_fs']:
        model = RandomForestClassifier()
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__class_weight': ['balanced']
        }
    elif model_type in ['xgboost', 'xgboost_no_fs']:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=None)
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__scale_pos_weight': [1, 10, 100, 1000]
        }
    elif model_type in ['svm', 'svm_no_fs']:
        model = SVC(probability=True)
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            'classifier__class_weight': ['balanced']
        }
    elif model_type == 'l1_logistic_regression':
        model = LogisticRegression(penalty='l1', solver='liblinear')
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__class_weight': ['balanced']
        }
    else:
        raise ValueError("Unsupported model type. Choose from 'logistic_regression', 'random_forest', 'xgboost', 'svm', 'l1_logistic_regression'.")
    
    return model, param_grid

# Generalized function to create and train the model pipeline with class weights
def train_model(X_train, y_train, model, param_grid, model_name, model_output_folder, feature_selection='False'):
    start_time = time.time()
    
    # Define the preprocessing for numeric features (imputation + scaling)
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    # Define the preprocessing for categorical features (one-hot encoding)
    categorical_features = X_train.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the feature selection step if not 'no_fs' or 'l1_logistic_regression'
    if feature_selection == 'True' and 'l1_logistic_regression' not in model_name:
        feature_selection = SelectPercentile(score_func=f_classif, percentile=1)
        pipeline_steps = [
            ('preprocessor', preprocessor),
            ('feature_selection', feature_selection),
            ('classifier', model)
        ]
    elif feature_selection == 'False' or 'l1_logistic_regression' in model_name:
        print(f"Skipping feature selection for {model_name}.")
        wandb.log({"status": f"Skipping feature selection for {model_name}"})
        pipeline_steps = [
            ('preprocessor', preprocessor),
            ('classifier', model)
        ]

    print("Starting model training...")
    wandb.log({"status": "Starting model training"})

  # Define the model pipeline
    pipeline = Pipeline(steps=pipeline_steps)

    # Perform 5-fold cross-validation with parallelization and verbose output
    grid_search_start_time = time.time()

    if 'xgboost' in model_name:
        n_threads = 1 # XGBoost does not support n_jobs parameter in conjunction with sklearn's GridSearchCV
    else:
        n_threads = -1

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=n_threads, verbose=10)
    grid_search.fit(X_train, y_train)
    grid_search_end_time = time.time()

    print("Model training completed.")
    wandb.log({"status": "Model training completed"})

    # Log the best parameters and best score to wandb
    wandb.log({"best_params": grid_search.best_params_, "best_score": grid_search.best_score_})

    # Print the best parameters and best score
    print(f"Best parameters found for {model.__class__.__name__}: ", grid_search.best_params_)
    print(f"Best cross-validation AUC for {model.__class__.__name__}: ", grid_search.best_score_)

    # Save the best estimator
    model_path = os.path.join(model_output_folder, f'{model_name}_best_model.pkl')
    joblib.dump(grid_search.best_estimator_, model_path)

    # Log the model to wandb
    wandb.save(f'{model_name}_best_model.pkl')

    # Log timing information
    total_time = time.time() - start_time
    grid_search_time = grid_search_end_time - grid_search_start_time
    wandb.log({"total_time": total_time, "grid_search_time": grid_search_time})
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Grid search time: {grid_search_time:.2f} seconds")

    # Log summary metrics
    wandb.summary['best_score'] = grid_search.best_score_
    wandb.summary['total_time'] = total_time
    wandb.summary['grid_search_time'] = grid_search_time

    # Return the best estimator
    return grid_search.best_estimator_

# Function to plot metrics
def plot_metrics(model, X, y, dataset_name, model_name, plot_output_folder):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    wandb.log({f"{dataset_name}_accuracy": accuracy})
    print(f"{dataset_name} set accuracy: ", accuracy)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset_name}')
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(plot_output_folder, f'{model_name}_roc_curve_{dataset_name}.png')
    plt.savefig(roc_curve_path)
    plt.close()
    wandb.log({f"{dataset_name}_roc_curve": wandb.Image(roc_curve_path)})

    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve - {dataset_name}')
    pr_curve_path = os.path.join(plot_output_folder, f'{model_name}_precision_recall_curve_{dataset_name}.png')
    plt.savefig(pr_curve_path)
    plt.close()
    wandb.log({f"{dataset_name}_precision_recall_curve": wandb.Image(pr_curve_path)})

    # Confusion matrix and F1 score at different thresholds
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred_threshold)
        f1 = f1_score(y, y_pred_threshold)
        wandb.log({f"{dataset_name}_confusion_matrix_threshold_{threshold}": cm, f"{dataset_name}_f1_score_threshold_{threshold}": f1})
        print(f"Confusion Matrix at threshold {threshold} - {dataset_name}:\n{cm}")
        print(f"F1 Score at threshold {threshold} - {dataset_name}: {f1}")
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix at threshold {threshold} - {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(plot_output_folder, f'{model_name}_confusion_matrix_{dataset_name}_threshold_{threshold}.png')
        plt.savefig(cm_path)
        plt.close()
        wandb.log({f"{dataset_name}_confusion_matrix_{threshold}": wandb.Image(cm_path)})

# # Function to plot retained features - Deprecated as functionality transferred to 2_Polyproteomic_FeatureImportance_Evaluator.py
# def plot_retained_features(model, X, feature_names, model_name, folder_path):
#     # Get the feature selection step from the pipeline
#     feature_selection = model.named_steps['feature_selection']
#     mask = feature_selection.get_support()  # Get the boolean mask of selected features
    
#     # Debugging: Print shapes of feature_names and mask
#     print(f"Shape of feature_names: {feature_names.shape}")
#     print(f"Shape of mask: {mask.shape}")

#     # Ensure the mask length matches the number of features
#     if len(mask) != len(feature_names):
#         raise ValueError("The length of the feature selection mask does not match the number of features.")

#     selected_features = feature_names[mask]
#     all_scores = feature_selection.scores_

#     # Sort the retained features by descending order of score magnitude
#     sorted_indices_retained = np.argsort(all_scores[mask])[::-1]
#     sorted_selected_features = selected_features[sorted_indices_retained]
#     sorted_selected_scores = all_scores[mask][sorted_indices_retained]

#     # Plot the retained features
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=sorted_selected_features, y=sorted_selected_scores)
#     plt.xticks(rotation=90)
#     plt.title('Retained Features after Model-Based Feature Selection')
#     plt.xlabel('Features')
#     plt.ylabel('ANOVA F-ratio')
#     retained_features_path = os.path.join(folder_path, f'{model_name}_retained_features.png')
#     plt.savefig(retained_features_path)
#     plt.close()
#     wandb.log({f"{model_name}_retained_features": wandb.Image(retained_features_path)})

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model for case-control classification.')
    parser.add_argument('--model', type=str, required=True, help="Model type: 'logistic_regression', 'random_forest', 'xgboost', 'svm'")
    parser.add_argument('--plot_output_folder', type=str, required=True, help="Folder path to save the plots and model")
    parser.add_argument('--model_output_folder', type=str, required=True, help="Folder path to save the model")
    parser.add_argument('--feature_selection', type=str, required=True, help="Define whether or not feature selection is applied prior to training")
    args = parser.parse_args()

    # Create the output folder if it does not exist
    if not os.path.exists(args.plot_output_folder):
        os.makedirs(args.plot_output_folder)

    # Initialize wandb with a project name based on the model type
    wandb.init(project=f"polyproteomic_casecontrol_{args.model}")

# Log configuration parameters
    wandb.config.update({
        "model_type": args.model,
        "plot_output_folder": args.plot_output_folder
    })

    print("Importing data...")
    wandb.log({"status": "Importing data"})

    # Import data
    exclusion_list = pd.read_csv('../DATA/UKB/w11223_20241217.csv', header=None)
    exclusion_list.columns=['eid'] #Rename the column to 'eid'
    pp_i0_covariates = pd.read_csv('../DATA/UKB/PROCESSED/2_5Y/ukb_ALLpanel_pp_hcm_covariates_bp_t2d_smoking_rcmmcov.tsv', sep='\t') #Already filtered for individuals which are at least non-NA in one Olink pp and non-NA values in covariates.

    print(exclusion_list.shape)
    print(pp_i0_covariates.shape)

    #Print the number of HCM cases and controls in pp_i0_covariates
    print(pp_i0_covariates['hcm'].value_counts())

    #Filter the pp and the covariate dataframes for individuals not in exclusion list
    pp_i0_covariates = pp_i0_covariates[~pp_i0_covariates['eid'].isin(exclusion_list['eid'])]
    print(pp_i0_covariates.shape) #49,586 individuals

    #Extract out the X and y variables from the pp_i0_covariates dataframe where the y variable is labelled 'hcm' column and the x variable is all other columns
    X = pp_i0_covariates.drop(columns=['eid', 'hcm', 'instance'])
    y = pp_i0_covariates['hcm']

    print("Data import completed.")
    wandb.log({"status": "Data import completed"})

    print("Splitting data into training and test sets...")
    wandb.log({"status": "Splitting data into training and test sets"})

    #Split off the X and y dataframes into a 80% training and 20% test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(X_train.shape)
    print(X_test.shape)

    #Print the number of HCM=true and HCM=false in the training and test set
    print(y_train.value_counts())
    print(y_test.value_counts())

    print("Data splitting completed.")
    wandb.log({"status": "Data splitting completed"})

    print("Saving training and test sets to CSV files...")
    wandb.log({"status": "Saving training and test sets to CSV files"})

    #Write out the training and test sets to csv files
    X_train.to_csv(os.path.join(args.output_folder,'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(args.output_folder,'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(args.output_folder,'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(args.output_folder,'y_test.csv'), index=False)

    print("CSV files saved.")
    wandb.log({"status": "CSV files saved"})

    # Get the model and parameter grid based on user input
    model, param_grid = get_model_and_params(args.model)

    # Log the parameter grid to wandb config
    wandb.config.update(param_grid)

    # Train the model
    trained_model = train_model(X_train, y_train, model, param_grid, args.model, args.model_output_folder, args.feature_selection)

    # Plot metrics for the training set
    plot_metrics(trained_model, X_train, y_train, "Training", args.model, args.plot_output_folder)

    # # Get feature names after preprocessing
    # preprocessor = trained_model.named_steps['preprocessor']
    # feature_names = np.concatenate([
    #     preprocessor.transformers_[0][1].named_steps['scaler'].get_feature_names_out(X_train.select_dtypes(include=['int64', 'float64']).columns),
    #     preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(X_train.select_dtypes(include=['object']).columns)
    # ])

    # Plot retained features for the final model on the training set
    # plot_retained_features(trained_model, X_train, feature_names, args.model, args.output_folder)