import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

test_df = pd.read_csv('test.csv')

input_cols = list(test_df.columns)[1:-1]
target_col = 'RainTomorrow'

test_inputs = test_df[input_cols].copy()
test_inputs.drop(columns='Date', inplace=True)
test_targets = test_df[target_col].copy()

# Numeric & Categorical Cols
numeric_cols = test_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = test_inputs.select_dtypes('object').columns.tolist()

print(test_inputs[numeric_cols].describe())
print(test_inputs[categorical_cols].nunique())

# Imputer

