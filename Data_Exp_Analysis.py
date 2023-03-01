import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']= (10,6)
matplotlib.rcParams['figure.facecolor']='#00000000'

raw_df = pd.read_csv('weatherAUS.csv')

print('Data Info:')
print(raw_df.info())
print('Statistical Description:')
print(raw_df.describe())

# drop null/NA values from rain today & rain tomorrow
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
print(raw_df.columns)


# Exploratory Data Analysis
# Exploring the distributions of various columns and see how they are related to each other & the target column.
print('Unique_Locations= ',raw_df['Location'].nunique())
fig = px.histogram(raw_df, x='Location', title='Location vs.Rainy Days', color='RainToday')
fig.show()

fig = px.histogram(raw_df, x='Temp3pm', title='Temperature at 3 pm vs. Rain Tomorrow', color='RainTomorrow')
fig.show()

fig = px.histogram(raw_df, x='RainTomorrow', title='Rain Tomorrow vs. Rain Today', color='RainToday')
fig.show()

fig = px.scatter(raw_df, x='MinTemp', y='MaxTemp', title='Min Temp. vs. Max Temp', color='RainTomorrow')
fig.show()

fig = px.scatter(raw_df, x='Temp3pm', y='Humidity3pm', title='Temp3pm vs. Humidity3pm', color='RainTomorrow')
fig.show()

fig = px.scatter(raw_df, x='Pressure3pm', y='Humidity3pm', title='pressure3pm vs. humidity3pm', color='RainTomorrow')
fig.show()

fig = px.scatter(raw_df, x='Cloud3pm', y='Temp3pm', title='cloud3pm vs temp3pm', color='RainTomorrow')
fig.show()

fig = px.scatter(raw_df, x='WindSpeed3pm', y='Cloud3pm', title='Wind 3pm vs Cloud 3pm', color='RainTomorrow')
fig.show()


# Training, Validation & Test Set
from sklearn.model_selection import train_test_split

train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

print('train_df:', train_df.shape)
print('val_df', val_df.shape)
print('test_df', test_df.shape)

# Seperating Train, Val & Test data with date & time:
## Train data < 2015
## Val Data == 2015
## Test Data > 2015

year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

print(' ')
print('Updated data sheets: ')

print('train_df:', train_df.shape)
print('val_df', val_df.shape)
print('test_df', test_df.shape)

train_df.to_csv('train.csv')
val_df.to_csv('val.csv')
test_df.to_csv('test.csv')