'''
Snakemake pipeline to iterate first-pass training in the training set over all model types of interest
'''

import os

configfile: "1_Polyproteomic_ML_Train.yaml"

rule all:
    input:
        model_pkl_files = expand(config['model_output_folder']+"{model_name}_best_model.pkl", model_name=config['model_names'])

rule train:
    input:
        X_train_data_path=config['X_train_data_path'],
        y_train_data_path=config['y_train_data_path']
    output:
        model_pkl_file = config['model_output_folder']+"{model_name}_best_model.pkl"
    threads: 8
    params:
        model_output_folder=config['model_output_folder'],
        plot_output_folder=lambda wildcards: config['model_output_folder'] + config['model_output_names'][config['model_names'].index(wildcards.model_name)],
        features_to_bypass=config['features_to_bypass'], #This refers to the quantitative features you should NOT pass to feature selection step
        features_to_select=config['features_to_select'],  #This refers to the quantitative features you SHOULD pass to feature selection step
        target_variable=config['target_variable'],
        feature_selection=lambda wildcards: config['feature_selection'][config['model_names'].index(wildcards.model_name)]
    shell:'''
        module load Miniforge3/24.1.2-0
        eval "$(conda shell.bash hook)"
        conda activate python3.11_ml

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