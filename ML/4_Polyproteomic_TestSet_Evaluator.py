import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score
from xgboost import XGBClassifier
from xgboost import DMatrix
from sklearn.pipeline import Pipeline
import sklearn
import numpy as np
from scipy.stats import bootstrap
import shap


sklearn.set_config(transform_output="pandas")

# Define the statistic function for AUC
def auc_statistic(y_true, y_pred_proba):
    return auc(*roc_curve(y_true, y_pred_proba)[:2])

def evaluate_model(model, X_test, y_test, plot_output_path, model_name, n_bootstraps=1000, random_seed=42):

    # Ensure the model includes preprocessing
    if not isinstance(model, Pipeline):
        raise ValueError("The model should be a Pipeline that includes preprocessing steps.")
    
    if 'xgboost' in model_name:
        # Perform the preprocessing directly on the X_test including KNN imputation for plasma proteins and feature selection to those used by the model itself
        X_test_preprocessed = model.named_steps['imputer_preprocessor'].fit_transform(X_test)
        model_features = model.named_steps['classifier'].get_booster().feature_names
        X_test_preprocessed = X_test_preprocessed[model_features]
        dtest = DMatrix(X_test_preprocessed)
        y_pred_proba = model.named_steps['classifier'].get_booster().predict(dtest)
        
        if model.named_steps['classifier'].n_classes_ == 2:
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # For multi-class, probabilities are returned per class
            # Take the class with highest probability
            y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        


    # Calculate metrics
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print(f'Balanced Accuracy: {balanced_accuracy}')
    print(f'F1 Score: {f1}')

    # Check lengths of y_test and y_pred_proba
    print(f'Length of y_test: {len(y_test)}')
    print(f'Length of y_pred_proba: {len(y_pred_proba)}')

    # Ensure lengths are consistent
    if len(y_test) != len(y_pred_proba):
        raise ValueError(f'Inconsistent lengths: y_test has {len(y_test)} samples, but y_pred_proba has {len(y_pred_proba)} samples.')

    # Convert y_test to numpy array
    y_test_np = np.array(y_test)

    # Bootstrapping for AUC confidence interval
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_test_np), len(y_test_np))
        if len(np.unique(y_test_np[indices])) < 2: #This tosses resamples where there is only one type of class e.g all cases or all controls
            continue
        score = auc(*roc_curve(y_test_np[indices], y_pred_proba[indices])[:2])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Compute the 95% confidence interval
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    # Calculate the mean AUC
    mean_auc = auc_statistic(y_test, y_pred_proba)

    print(f'AUC: {mean_auc:.2f}')
    print(f'95% CI for AUC: [{ci_lower:.2f}, {ci_upper:.2f}]')
    
    # Plot confusion matrix at various decision thresholds
    thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_threshold)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix at Threshold {threshold}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{plot_output_path}/{model_name}_confusion_matrix_{threshold}.png')
        plt.close()
    
    # Plot precision-recall curve and add a label to show the average_precision score (i.e PR-AUC)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    average_precision = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'AP = {average_precision:.2f}')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'{plot_output_path}/{model_name}_precision_recall_curve.png')
    plt.close()

    # Plot ROC curve with AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label=f'AUC = {roc_auc:.2f} (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])')
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'{plot_output_path}/{model_name}_roc_curve.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a saved model on a test set and output performance plots.')
    parser.add_argument('--model_pkl_file', type=str, required=True, help='Path to the saved model .pkl file')
    parser.add_argument('--X_test_path', type=str, required=True, help='Path to the X_test.csv file')
    parser.add_argument('--y_test_path', type=str, required=True, help='Path to the y_test.csv file')
    parser.add_argument('--plot_output_path', type=str, required=True, help='Path to save the output plots')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model for plot titles and filenames') 
    parser.add_argument('--n_bootstraps', type=int, default=1000, help='Number of bootstrap samples for AUC confidence interval')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for bootstrapping')
    args = parser.parse_args()

    # Load the model
    model = joblib.load(args.model_pkl_file)
    X_test = pd.read_csv(args.X_test_path)

    if 'eid' in X_test.columns:
        X_test = X_test.drop(columns='eid')
    
    y_test = pd.read_csv(args.y_test_path).values.ravel()
    
    evaluate_model(model, X_test, y_test, args.plot_output_path, args.model_name, args.n_bootstraps, args.random_seed)