import pandas as pd

data=pd.read_csv("./data/data.csv", sep =";")
data.info()

X = data.drop('K', axis = 1)
y = data.K

# print(X.sample(5))

