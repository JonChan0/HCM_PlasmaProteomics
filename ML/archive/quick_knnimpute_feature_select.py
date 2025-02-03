import pandas as pd
from sklearn.impute import KNNImputer
import joblib
from sklearn.compose import ColumnTransformer
import sklearn

sklearn.set_config(transform_output="pandas")

X_train = pd.read_csv('../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_train.csv')
X_test = pd.read_csv('../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_test.csv')

X_all = pd.concat([X_train, X_test])

pp_names = pd.read_csv('../DATA/UKB/ML/2_covariates_pp/ML_pp_names.csv').loc[:, 'name'].tolist()

imputer = ColumnTransformer([('knn_imputer', KNNImputer(n_neighbors=5) ,pp_names)],verbose_feature_names_out=False)

X_all_ppimputed = imputer.fit_transform(X_all)
X_all_ppimputed.head()

X_all_ppimputed.to_csv('../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_all_ppimputed.csv', index=False)

model = joblib.load('../OUTPUT/UKB/ML/2_models/1_hcm_cc_noprs/xgboost_best_model.pkl')

preprocessor2 = model.named_steps['feature_preprocessor']
feature_names = preprocessor2.get_feature_names_out()

#Filter the X_all_ppimputed for the features in feature_names

X_all_ppimputed_feature_selected = X_all_ppimputed[['eid',feature_names]]
X_all_ppimputed_feature_selected.to_csv('../OUTPUT/UKB/ML/1_data/1_hcm_cc_noprs/X_all_ppimputed_feature_selected.csv', index=False)
