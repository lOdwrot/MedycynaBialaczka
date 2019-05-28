import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stats import calculateStatsForKMeans, calculateStatsForNM
from dataLoad import getDataSetLeukemia, getDataSetIris

# Load Data (param whether data should be normalized)
data = getDataSetLeukemia(False)
# data = getDataSetIris()
# K - class
X = data.drop('K', axis = 1)
y = data.K

# Build features ranking
test = SelectKBest(score_func=chi2, k=2)
test.fit(X, y)

scores = []
num_features = len(X.columns)
for i in range(num_features):
    score = test.scores_[i]
    scores.append((score, X.columns[i]))

sortedFeatures = sorted(scores, reverse = True)
sortedFeatures = list(map(lambda x: x[1], sortedFeatures))
print ('Features Ranking', sortedFeatures)

# Count stats for params
for i in range (1, len(sortedFeatures) + 1):
    subData = data.copy()
    for dropedFeatureIndex in range (i, len(sortedFeatures)):
        subData = subData.drop(sortedFeatures[dropedFeatureIndex], axis=1)
    print('Stats for features: ', i)
    # Print stats For NM
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForNM(subData, 'euclidean', True)
    # Print stats for 1-NN
    # accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(subData, 1, 'euclidean')
    # # Print stats for 10-NN
    # accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(subData, 10, 'euclidean')
    # # Print stats for 20-NN
    # accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(subData, 20, 'euclidean')
    print('Stats', accuracy, precision, recall, fscore)
    print('Confusion Matrix')
    print(confusionMatrix)

