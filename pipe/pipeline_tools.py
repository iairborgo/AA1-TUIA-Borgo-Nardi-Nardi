import pandas as pd
from math import sin, cos, pi
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer

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
    def __init__(self,cols, method='yeo-johnson', standardize=True, copy = True):
      self.cols = cols
      self.method = method
      self.standardize = standardize
      self.scaler = None
      self.copy = copy


    def fit(self, X, y=None):
      self.scaler = PowerTransformer(method=self.method, standardize=self.standardize, copy=self.copy)
      self.scaler.fit(X[self.cols])
      return self
      
    def transform(self, X):
      df = X.copy()
      df[self.cols] = self.scaler.transform(df[self.cols])
      return df

class CustomKNNImputer(BaseEstimator, TransformerMixin):
  def __init__(self, cols, n_neighbors):
    self.cols = cols
    self.n_neighbors = n_neighbors
    self.imputer = None
      
  def fit(self, X, y=None):
    self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
    self.imputer.fit(X[self.cols])
    return self
  
  def transform(self, X):
     df = X.copy()
     df[self.cols] = self.imputer.transform(df[self.cols])
     return df

class PredictDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X.iloc[idx].values

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
    if pd.isna(row['Cloud9am']) and pd.isna(row['Cloud3pm']):
        row['Cloud9am'] = 1 if row['RainToday'] == 0 else 7
        row['Cloud3pm'] = 1 if row['RainToday'] == 0 else 7
    elif pd.isnull(row['Cloud9am']):
        row['Cloud9am'] = row['Cloud3pm']
    elif pd.isna(row['Cloud3pm']):
        row['Cloud3pm'] = row['Cloud9am']
    return row

def impute_scale_cloud(X):
  X = X.copy()
  X = X.apply(impute_cloud_toapply, axis = 1)
  X['Cloud9am'] /= 8
  X['Cloud3pm'] /= 8
  return X

def debug(X):
   print(X)
   return X