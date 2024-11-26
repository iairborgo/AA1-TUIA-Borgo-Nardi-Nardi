import pandas as pd
import random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline, FunctionTransformer
from pipeline_tools import *
from skorch import NeuralNetBinaryClassifier
import torch.optim as optim
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

model = NeuralNetBinaryClassifier(
    RainClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.008458413004356744,
    max_epochs=50,
    batch_size=128,
    verbose=False,
    train_split=None
)
model.initialize()

transform_pipeline = Pipeline(steps=[
    ('binarize_rain', BinaryEncoder('RainToday')),
    ('week_encoder', WeekTrigonometricEncoder('Date', drop_initial=True)),
    ('windgust_encoder', FunctionTransformer(WindGustTrigonometricEncoder)),
    ('clip', FunctionTransformer(clip, kw_args={'col': 'Cloud9am', 'minval': 0, 'maxval': 8, 'fill': 0})),
    ('impute_scale_cloud', FunctionTransformer(impute_scale_cloud)),
    ('drop', FunctionTransformer(drop_columns, kw_args={'columns': drop})),
    ('custom_power_transform', CustomPowerTransformer(cols = power_cols)), 
    ('print_debug', FunctionTransformer(debug)),
    ('knn_imputer', KNNImputer(n_neighbors=155)),
    ('float32', FunctionTransformer(func=lambda X: torch.tensor(X, dtype=torch.float32).numpy(), validate=False)),
    ('model', model)
])

y_train = y_train.apply(lambda X: 1 if X=='Yes' else 0)
#y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
transform_pipeline.fit(X_train, y_train)


y_test = y_test.apply(lambda X: 1 if X=='Yes' else 0)
y_test.sum()
from sklearn.metrics import classification_report
y_pred = transform_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))