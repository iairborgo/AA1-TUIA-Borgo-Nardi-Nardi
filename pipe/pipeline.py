import pandas as pd
import random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline, FunctionTransformer
from pipe.pipeline_tools import *
import joblib

df = pd.read_csv('weatherAUS.csv')
df[df['Location'] == 'MountGini']['Location'] = 'MountGinini'

random.seed(147855)

selected_locations = random.sample(list(df['Location'].unique()), 10)

df = df[df['Location'].isin(selected_locations)]

df = df.dropna(subset=['RainTomorrow'])
df.drop(df[df.isnull().sum(axis=1) > 11].index, inplace=True)

y = df['RainTomorrow']
X = df.drop(columns = ['RainTomorrow'])

ros = RandomOverSampler(random_state=738)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
X_train, y_train = ros.fit_resample(X_train, y_train)

drop = ['Location', 'WindGustDir', 'Temp3pm', 'Temp9am', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm', 'WindDir9am', 'WindDir3pm', 'Pressure9am']
power_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'WindGustSpeed','Humidity9am', 'Humidity3pm', 'Pressure3pm']
knn_cols = power_cols + ['sin_week', 'cos_week', 'WindGustDir_cos','WindGustDir_sin']

transform_pipeline = Pipeline(steps=[
    ('binarize_rain', BinaryEncoder('RainToday')),
    ('week_encoder', WeekTrigonometricEncoder('Date', drop_initial=True)),
    ('windgust_encoder', FunctionTransformer(WindGustTrigonometricEncoder)),
    ('clip', FunctionTransformer(clip, kw_args={'col': 'Cloud9am', 'minval': 0, 'maxval': 8, 'fill': 0})),
    ('impute_scale_cloud', FunctionTransformer(impute_scale_cloud)),
    ('drop', FunctionTransformer(drop_columns, kw_args={'columns': drop})),
    ('custom_power_transform', CustomPowerTransformer(cols = power_cols)), 
    ('knn_imputer', CustomKNNImputer(n_neighbors=155, cols = knn_cols)),
])

transform_pipeline.fit(X_train)

joblib.dump(transform_pipeline, 'pipeline_final.pkl')