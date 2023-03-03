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
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
df2 = df[categorical_cols].fillna('unknown')
encoder.fit(df2)
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

print(train_inputs)

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

print(train_inputs)

# Removing unnecessary columns
train_inputs.drop(columns=categorical_cols, inplace=True)
val_inputs.drop(columns=categorical_cols, inplace=True)

# Model Building
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)
print(model.coef_.tolist())
print(model.intercept_)

# Generating Predictions
train_predictions = model.predict(train_inputs)
print('Train Predictions: ',train_predictions)

# Accuracy Score
from sklearn.metrics import accuracy_score
ast =accuracy_score(train_targets,train_predictions)
print('Train Accuracy Score: ', ast)

# Checking on Validation Test set
val_predictions = model.predict(val_inputs)
print('Val Predictions:',val_predictions)
print('Val Accuracy Score: ',accuracy_score(val_targets, val_predictions))

print('CLasses are:', model.classes_)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
#----Train df
cmt =confusion_matrix(train_targets, train_predictions, normalize='true')
print('Train Confusion Matrix: ', cmt)

#----Val df
cmt2 = confusion_matrix(val_targets,val_predictions, normalize='true')
print('Val Confusion Matrix:', cmt2)

# Saving all the models & accessories
import joblib
australia_rain = {
    'model': model,
    'imputer':imputer,
    'scaler':scaler,
    'encoder':encoder,
    'input_cols':input_cols,
    'target_col':target_col,
    'numeric_cols':numeric_cols,
    'categorical_cols':categorical_cols,
    'encoded_cols':encoded_cols
}
joblib.dump(australia_rain, 'australia_rain.joblib')

