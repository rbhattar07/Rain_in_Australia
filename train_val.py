import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

df = pd.read_csv('weatherAUS.csv')
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
imputer.fit(df[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])

imputer.fit(df[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])

# Scaling numeric Features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])

scaler.fit(df[numeric_cols])
val_inputs[numeric_cols]=scaler.transform(val_inputs[numeric_cols])

print(train_inputs[categorical_cols])
print(' ')
# Encoding Categorical Features
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
df2 = df[categorical_cols].fillna('unknown')
encoder.fit(df2)
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

print(train_inputs)

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

print(train_inputs)
