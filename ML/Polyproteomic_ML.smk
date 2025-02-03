'''
Snakemake pipeline for investigating ML models for using plasma proteomic data for predictive utility
Author: Jonathan Chan
Date: 2025-01-23
'''

import os

configfile: "config.yaml"

rule all:
    input:
        X_train_data_path=config['base_data_folder'] + config['folder_suffix'] + "X_train.csv",
        X_test_data_path=config['base_data_folder'] + config['folder_suffix'] + "X_test.csv",
        y_train_data_path=config['base_data_folder'] + config['folder_suffix'] + "y_train.csv",
        y_test_data_path=config['base_data_folder'] + config['folder_suffix']+ "y_test.csv",
        model_pkl_files = expand(config['base_model_folder'] + config['folder_suffix']+"{model_name}_best_model.pkl", model_name=config['model_names'])

#This outlines the splitting of the original dataset into X_train and X_test etc.
rule data_splitting:
    input:
        data_path = config['data_path']
    output:
        X_train_data_path=config['base_data_folder'] + config['folder_suffix'] + "X_train.csv",
        X_test_data_path=config['base_data_folder'] + config['folder_suffix'] + "X_test.csv",
        y_train_data_path=config['base_data_folder'] + config['folder_suffix'] + "y_train.csv",
        y_test_data_path=config['base_data_folder'] + config['folder_suffix']+ "y_test.csv"
    params:
        data_output_folder = config['base_data_folder'] + config['folder_suffix'],
        target_variable = config['target_variable']
    conda: 'python3.11_ml'
    threads: 2
    resources:
        mem_mb = lambda wildcards, threads: threads * 4000  # Example: 4000 MB per thread
    shell:'''
        # Set the number of threads for Python
        export OMP_NUM_THREADS=2

        python 0_Polyproteomic_Data_Splitter.py \
            --data_path {input.data_path} \
            --data_output_folder {params.data_output_folder} \
            --target_variable = {params.target_variable}
    '''

#This outlines the first-pass training over the various different model types to compare different model classes 
rule train:
    input:
        X_train_data_path=rules.data_splitting.output.X_train_data_path,
        y_train_data_path=rules.data_splitting.output.y_train_data_path
    output:
        model_pkl_file = config['base_model_folder'] + config['folder_suffix']+"{model_name}_best_model.pkl"
    threads: 8
    conda: 'python3.11_ml'
    resources:
        mem_mb = lambda wildcards, threads: threads * 4000  # Example: 4000 MB per thread
    params:
        model_output_folder=config['base_model_folder'] + config['folder_suffix'],
        plot_output_folder=lambda wildcards: config['base_model_folder'] + config['folder_suffix'] + config['model_output_names'][config['model_names'].index(wildcards.model_name)],
        features_to_bypass=config['features_to_bypass'], #This refers to the quantitative features you should NOT pass to feature selection step
        features_to_select=config['features_to_select'],  #This refers to the quantitative features you SHOULD pass to feature selection step
        target_variable=config['target_variable'],
        feature_selection=lambda wildcards: config['feature_selection'][config['model_names'].index(wildcards.model_name)]
    shell:'''
        # Set the number of threads for Python
        export OMP_NUM_THREADS=8

        python 1_Polyproteomic_ML_Train.py \
        --model {wildcards.model_name} \
        --plot_output_folder "{params.plot_output_folder}" \
        --model_output_folder "{params.model_output_folder}" \
        --feature_selection {params.feature_selection} \
        --features_to_bypass_fs {params.features_to_bypass} --features_to_select_fs {params.features_to_select} \
        --target_variable {params.target_variable} \
        --X_train_data {input.X_train_data_path} --y_train_data {input.y_train_data_path}
    '''

#This outlines the feature importance plotting for each model type

rule feature_importance_plotter:
    input:
        model_pkl_file = rules.train.output.model_pkl_file,
        X_preprocessed_data_path = config['base_data_folder'] + config['folder_suffix'] + 'X_train_preprocessed_{model_name}.csv'
    output:
        feature_importance_plot = config['base_plot_folder'] + config['folder_suffix'] + 'feature_importance/{wildcards.model_name}_feature_importances.png'
    conda: 'python3.11_ml'
    threads: 2
    resources:
        mem_mb = lambda wildcards, threads: threads * 4000  # Example: 4000 MB per thread
    params:
        output_folder =  config['base_plot_folder'] + config['folder_suffix'] + 'feature_importance/',
    shell:'''
        module load Miniforge3/24.1.2-0
        eval "$(conda shell.bash hook)"
        conda activate python3.11_ml

        python 2_Polyproteomic_FeatureImportance_Evaluator.py  \
            --model_name {wildcards.model_name} \
            --model_pkl_file {input.model_pkl_file} \
            --X_data_file {input.X_preprocessed_data_path} \
            --output_folder {params.output_folder}
    '''
