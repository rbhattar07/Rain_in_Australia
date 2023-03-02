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
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(train_df[numeric_cols])
train_df[numeric_cols] = imputer.transform(train_df[numeric_cols])

imputer.fit(val_df[numeric_cols])
val_df[numeric_cols] = imputer.transform(val_df[numeric_cols])

# Scaling numeric Features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_df[numeric_cols])
train_df[numeric_cols]=scaler.transform(train_df[numeric_cols])

scaler.fit(val_df[numeric_cols])
val_df[numeric_cols]=scaler.transform(val_df[numeric_cols])