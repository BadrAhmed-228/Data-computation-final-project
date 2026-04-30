import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.svm import SVC

SEED = 42

# Feature engineering function
def feature_engineering(df):
    df = df.copy()
    df['capital_net'] = df['capital_gain'] - df['capital_loss']
    df['has_capital_gain'] = (df['capital_gain'] > 0).astype(int)
    df['has_capital_loss'] = (df['capital_loss'] > 0).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100],
                             labels=['young', 'early_career', 'mid_career', 'senior', 'retired'])
    df['hours_group'] = pd.cut(df['hours_per_week'], bins=[0, 20, 40, 60, 100],
                               labels=['part_time', 'full_time', 'overtime', 'extreme'])
    df['is_married'] = df['marital_status'].isin(['Married-civ-spouse', 'Married-AF-spouse']).astype(int)
    df['is_us_native'] = (df['native_country'] == 'United-States').astype(int)
    df['high_edu'] = (df['education_num'] >= 13).astype(int)
    return df

# Columns as in notebook
num_cols = [
    'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week',
    'capital_net', 'has_capital_gain', 'has_capital_loss', 'is_married', 'is_us_native', 'high_edu'
]
cat_cols = [
    'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'native_country', 'age_group', 'hours_group'
]

def build_pipeline(n_components_pca=12, k_best=18):

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),

        ('variance_filter', VarianceThreshold(threshold=0.01)),

        # IMPORTANT FIX: ensure mutual_info_classif is imported
        ('mi_filter', SelectKBest(score_func=mutual_info_classif, k=k_best)),

        ('pca', PCA(n_components=n_components_pca, random_state=SEED)),

        ('svm', SVC(
            gamma='scale',
            C=0.1,
            kernel='rbf',
            probability=True,
            random_state=SEED
        ))
    ])

    return pipeline

def save_model(model, path='model.pkl'):
    joblib.dump(model, path)

def load_model(path='model.pkl'):
    return joblib.load(path)
