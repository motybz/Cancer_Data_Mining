import os , csv
from sklearn import datasets
import numpy as np
import pandas as pd

def load_dataset(source_file = ''):
    with open(source_file,"r") as f:
        data = f.readlines()[1]
        data = data.split(',')
        ncols = len(data)
    # dtype=None to handle dirty data
    df = pd.read_csv(source_file, header=0)
    original_headers = list(df.columns.values)
    Y_set = df.iloc[:,-1]
    # Y_set = np.genfromtxt(source_file, delimiter=",", skip_header=1, usecols= ncols -1 , dtype=None)
    # print("The Y colunm is: ",Y_set)
    # index_set = np.genfromtxt(source_file, delimiter=",", skip_header=1, usecols= 0 , dtype=None)
    # print("The id of each row is: ", index_set)
    X_set = df.iloc[:,:-1]
    # print("The first row is: ", X_set[0])
    return X_set , Y_set ,original_headers