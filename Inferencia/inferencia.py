import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # para anadir pipe.

import warnings
warnings.filterwarnings('ignore')

import argparse
from pipe.pipeline_tools import *
import time
import pandas as pd
import joblib
from torch.utils.data import DataLoader


device = 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type = str)

args = parser.parse_args()

print(args.file)
try:
    df = pd.read_csv(args.file)
    print('Data Ingresada.')
except:
    print('Error encontrando el archivo.')
    time.sleep(3)
    sys.exit()
        
transform_pipeline = joblib.load("pipeline_final.pkl")
model = torch.jit.load("model0_optuna.pt",map_location=torch.device(device)).eval()

df = transform_pipeline.transform(df)
df = PredictDataset(df)
df_dataloader = DataLoader(df, batch_size = 180, shuffle = False)

y_pred_nn = []
with torch.no_grad():
  for X in df_dataloader:
    X = torch.tensor(X, dtype=torch.float32)
    X = X.to(device)
    pred = model(X)
    pred_sigmoid = torch.sigmoid(pred)
    bin = [pred_sigmoid > 0.5][0].tolist()
    bin = [int(x[0]) for x in bin]
    y_pred_nn.extend(bin)

y_pred_nn = pd.DataFrame(y_pred_nn)
y_pred_nn.to_csv('/app/data/predict.csv', index = False)

print('Predicciones guardadas en predict.csv')
time.sleep(3)
