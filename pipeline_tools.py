import pandas as pd
from math import sin, cos, pi
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
from sklearn.preprocessing import PowerTransformer

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column, drop_initial = False):
        self.column = column
        self.drop_initial = drop_initial

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].apply(lambda x: 1 if x == 'Yes' else 0)
        return X

class WeekTrigonometricEncoder(BaseEstimator, TransformerMixin):
  def __init__(self, column, drop_initial = False):
    self.column = column
    self.drop_initial = drop_initial

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()
    if X[self.column].dtype == 'object':
      X[self.column] = pd.to_datetime(X[self.column])
    X.loc[:, 'week'] = X[self.column].dt.isocalendar().week
    X.loc[:, 'week'] = X['week'].astype(float) / 53

    X[f'sin_week'] = X['week'].apply(lambda x: sin(2 * pi * x))
    X[f'cos_week'] = X['week'].apply(lambda x: cos(2 * pi * x))

    X.drop('week', axis = 1, inplace = True)

    if self.drop_initial:
      X.drop(self.column, axis = 1, inplace = True)

    return X
  
class CustomPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,cols, method='yeo-johnson', standardize=True):
      self.cols = cols
      self.method = method
      self.standardize = standardize
      self.scaler = None

    def fit(self, X, y=None):
      self.scaler = PowerTransformer(method=self.method, standardize=self.standardize, copy=self.copy)
      self.scaler.fit(X)
      
    def transform(self, X):
      df = X.copy()
      df[self.cols] = self.scaler.transform(df[self.cols])
      return df

def drop_columns(X, columns):
    X = X.copy()
    return X.drop(columns, axis=1)

def impute(X, col, val):
    X = X.copy()
    X[col] = X[col].fillna(val)
    return X

def clip(X, col, maxval, minval, fill):
   X.loc[X[col] > maxval] = fill
   X.loc[X[col] < minval] = fill
   return X

def WindGustTrigonometricEncoder(X):
    df = X.copy()
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    df['WindGustDir'] = df['WindGustDir'].apply(lambda x: directions.index(x) if x in directions else None)
 
    df['WindGustDir_cos'] = df['WindGustDir'].apply(lambda x: cos(2 * pi * x / 16) if x is not None else None)
    df['WindGustDir_sin'] = df['WindGustDir'].apply(lambda x: sin(2 * pi * x / 16) if x is not None else None)

    df.drop(columns = ['WindGustDir'], axis = 1)
    return df

def impute_cloud_toapply(row):
    if pd.isnull(row['Cloud9am']) and pd.isnull(row['Cloud3pm']):
        row['Cloud9am'] = 1 if row['RainToday'] == 0 else 7
        row['Cloud3pm'] = 1 if row['RainToday'] == 0 else 7
    elif pd.isnull(row['Cloud9am']):
        row['Cloud9am'] = row['Cloud3pm']
    else:
        row['Cloud3pm'] = row['Cloud9am']
    return row

def impute_scale_cloud(X):
  X = X.copy()
  X = X.apply(impute_cloud_toapply, axis = 1)
  X['Cloud9am'] /= 8
  X['Cloud3pm'] /= 8

'''
best_params = {
    "n_hidden": 4,
    "n_units_l0": 4,
    "dropout_l0": 0.4398398657839898,
    "n_units_l1": 12,
    "dropout_l1": 0.4571336631307156,
    "n_units_l2": 25,
    "dropout_l2": 0.08266737664565811,
    "n_units_l3": 9,
    "dropout_l3": 0.14446723137884201,
    "lr": 0.008458413004356744
}
'''

class RainClassifier(nn.Module):
    def __init__(self):
      super().__init__()
      
      self.lin1 = nn.Linear(in_features = 15, out_features = 4)
      self.relu1 = nn.ReLU()
      self.drop1 = nn.Dropout(0.45)
      self.lin2 = nn.Linear(in_features = 4, out_features = 12)
      self.relu2 = nn.ReLU()
      self.drop2 = nn.Dropout(0.45)
      self.lin3 = nn.Linear(in_features = 12, out_features = 25)
      self.relu3 = nn.ReLU()
      self.drop3 = nn.Dropout(0.085)
      self.lin4 = nn.Linear(in_features = 25, out_features = 9)
      self.relu4 = nn.ReLU()
      self.drop4 = nn.Dropout(0.145)
      self.lin5 = nn.Linear(in_features = 9, out_features = 1)

    def forward(self, x):
      x = self.lin1(x)
      x = self.relu1(x)
      x = self.drop1(x)
      x = self.lin2(x)
      x = self.relu2(x)
      x = self.drop2(x)
      x = self.lin3(x)
      x = self.relu3(x)
      x = self.drop3(x)
      x = self.lin4(x)
      x = self.relu4(x)
      x = self.drop4(x)
      x = self.lin5(x)
      return x