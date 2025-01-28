import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

parser = argparse.ArgumentParser(description='Split a dataset using 80/20 split.')
parser.add_argument('--data_path', type=str, default='', help="The model data containing both X and Y")
parser.add_argument('--data_output_folder', type=str, required=True, help="Folder path to save the plots and model")
parser.add_argument('--target_variable', type=str, default='', help="The target variable is either prevalent (i.e HCM case/control status) or incident (i.e incident HCM diagnosis)")
args = parser.parse_args()

# Create the output folder if it does not exist
if not os.path.exists(args.data_output_folder):
    os.makedirs(args.data_output_folder)

print("Importing data...")

# Import data
exclusion_list = pd.read_csv('../DATA/UKB/w11223_20241217.csv', header=None)
exclusion_list.columns=['eid'] #Rename the column to 'eid'
pp_i0_covariates = pd.read_csv(args.data_path, sep='\t') #Already filtered for individuals which are at least non-NA in one Olink pp and non-NA values in covariates.

print(exclusion_list.shape)
print(pp_i0_covariates.shape)

#Filter the pp and the covariate dataframes for individuals not in exclusion list
pp_i0_covariates = pp_i0_covariates[~pp_i0_covariates['eid'].isin(exclusion_list['eid'])]
print(pp_i0_covariates.shape) 

#Extract out the X and y variables from the pp_i0_covariates dataframe where the y variable is labelled 'hcm' column and the x variable is all other columns
if args.target_variable == 'prevalent':
    #Print the number of HCM cases and controls in pp_i0_covariates after exclusion
    print(pp_i0_covariates['hcm'].value_counts())

    X = pp_i0_covariates.drop(columns=['hcm', 'instance'])
    y = pp_i0_covariates.loc[:,'hcm']

elif args.target_variable == 'incident':
    #Print the number of HCM cases and controls in pp_i0_covariates after exclusion
    print(pp_i0_covariates['incidenthcm_status'].value_counts())
    X = pp_i0_covariates.drop(columns=['incidenthcm_status', 'instance', 'lost_age', 'death_age','age_attend_i0','dob_approx','min_diag_date2', 'datasetversion_age','incidenthcm_age'])
    y = pp_i0_covariates.loc[:,'incidenthcm_status']

print("Data import completed.")

print("Splitting data into training and test sets...")

#Split off the X and y dataframes into a 80% training and 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(X_train.shape)
print(X_test.shape)

#Print the number of HCM=true and HCM=false in the training and test set
print(y_train.value_counts())
print(y_test.value_counts())

print("Data splitting completed.")

print("Saving training and test sets to CSV files...")

#Write out the training and test sets to csv files
X_train.to_csv(os.path.join(args.data_output_folder,'X_train.csv'), index=False)
X_test.to_csv(os.path.join(args.data_output_folder,'X_test.csv'), index=False)
y_train.to_csv(os.path.join(args.data_output_folder,'y_train.csv'), index=False)
y_test.to_csv(os.path.join(args.data_output_folder,'y_test.csv'), index=False)

print("CSV files saved.")