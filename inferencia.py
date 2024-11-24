import argparse
#from pipeline_tools import *
import os
import time
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type = str)

args = parser.parse_args()
file = args.file

try:
    df = pd.read_csv(file)
    print('god no')
except:
    print('no god')
time.sleep(5)
