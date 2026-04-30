# In a script or notebook cell
import pandas as pd
from preprocess import feature_engineering, build_pipeline, save_model

# Load data as in notebook
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
COL_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]
df = pd.read_csv(URL, header=None, names=COL_NAMES, na_values=' ?', skipinitialspace=True)
df = feature_engineering(df)
df['income'] = (df['income'].str.strip().str.replace('.', '', regex=False) == '>50K').astype(int)
X = df.drop(columns=['income'])
y = df['income']

# Build and fit pipeline
pipeline = build_pipeline()
pipeline.fit(X, y)

# Save model
save_model(pipeline, 'model.pkl')