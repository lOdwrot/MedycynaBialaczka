import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

def getDataSetLeukemia(normalized=False):
    url = 'https://raw.githubusercontent.com/lOdwrot/MedycynaBialaczka/master/data/data.csv'
    data = pd.read_csv(url, sep =";", dtype='intc')
    data.columns = ['K', 'Temperatura', 'Anemia', 'Stopień krwawienia', 'Miejsce krwawienia', 'Ból kości', 'Wrażliwość mostka', 'Powiększenie węzłów chłonnych', 'Powiększenie wątroby', 'Centralny układ nerwowy', 'Powiększenie jąder', 'Uszkodzenie w sercu, płucach, nerce', 'Gałka oczna', 'Poziom WBC', 'Obniżenie poziomu RBC', 'Liczba płytek krwi', 'Niedojrzałe komórki', 'Stan pobudzenia szpiku', 'Główne komórki szpiku', 'Poziom limfocytów', 'Reakcja']
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

