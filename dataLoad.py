import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

def getDataSetLeukemia(normalized=False):
    url = 'https://raw.githubusercontent.com/lOdwrot/MedycynaBialaczka/master/data/data.csv'
    data = pd.read_csv(url, sep =";", dtype='float64')
    cols = data.columns.tolist()
    cols = cols[1:] + cols[:1]
    data = data[cols]
    features = list(filter(lambda x: x != 'K', data.columns))
    if normalized == True:
        data[features] = data[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return data

def getDataSetIris():
    iris = load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])
    data = data.rename(index=str, columns={"target": "K"})
    return data

