import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

test_df = pd.read_csv('test.csv')

# Importing the models
import joblib
joblib.load('australia_rain.joblib')

jb_ar_lib = joblib.load('australia_rain.joblib')

# defining input & target columns
input2 = jb_ar_lib['input_cols']
target2 = jb_ar_lib['target_col']
#Defining numeric, categorical & encoded columns
numeric_cols2 = jb_ar_lib['numeric_cols']
cat_cols2 = jb_ar_lib['categorical_cols']
encoded_cols2 = jb_ar_lib['encoded_cols']

# imputing missing numeric data
imputer2 = jb_ar_lib['imputer']
test_df[numeric_cols2] = imputer2.transform(test_df[numeric_cols2])
print(test_df[numeric_cols2])

# Scaling numeric data
scaler2 = jb_ar_lib['scaler']
test_df[numeric_cols2]=scaler2.transform(test_df[numeric_cols2])
print(test_df[numeric_cols2])

# Encoding Categorical columns
encoder2 = jb_ar_lib['encoder']
test_df[encoded_cols2] = encoder2.transform(test_df[cat_cols2])
print(test_df[cat_cols2])
print(test_df[encoded_cols2])