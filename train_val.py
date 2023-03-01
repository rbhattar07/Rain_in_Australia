import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')

input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'

train_inputs = train_df[input_cols].copy()
train_inputs.drop(columns='Date', inplace=True)
train_targets = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_inputs.drop(columns='Date', inplace=True)
val_targets = val_df[target_col].copy()

# Numeric & Categorical Cols
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

print(train_inputs[numeric_cols].describe())
print(train_inputs[categorical_cols].nunique())

#####

# Imputing Missing Numeric Data
