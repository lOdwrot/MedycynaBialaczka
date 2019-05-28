from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import confusion_matrix
import numpy as np

def countStats(_y_true, _y_pred):
  accuracy = accuracy_score(_y_true, _y_pred, normalize=True)
  precision = precision_score(_y_true, _y_pred, average='weighted')
  recall = recall_score(_y_true, _y_pred, average='weighted')
  fscore = f1_score(_y_true, _y_pred, average='weighted')
  
  return accuracy, precision, recall, fscore

def calculateStatsForKMeans(data, k=2, metric='euclidean'):
  tAccuracy = 0
  tPrecision = 0
  tRecall = 0
  tFScore = 0

  # Data used to draw confusion matrix will be gathered here
  ConfusionMatrixY = []
  ConfusionMatrixYpred = []
  
  for i in range(0, 5):
    kf = KFold(n_splits = 2, shuffle = True)
    result = next(kf.split(data), None)
    
    train1 = data.iloc[result[0]]
    test1 =  data.iloc[result[1]]
    train2 = data.iloc[result[1]]
    test2 =  data.iloc[result[0]]
    
    # Check first fold
    neigh = KNeighborsClassifier(n_neighbors=k, metric=metric)
    Xtrain = train1.drop('K', axis = 1)
    Ytrain = train1.K
    neigh.fit(Xtrain, Ytrain)
    
    Xtest = test1.drop('K', axis = 1)
    Ytest = test1.K
    YPred = neigh.predict(Xtest)

    if i == 0:
        ConfusionMatrixY = Ytest.tolist()
        ConfusionMatrixYpred = YPred.tolist()
    
    accuracy, precision, recall, fscore = countStats(Ytest, YPred)
    tAccuracy += accuracy
    tPrecision += precision
    tRecall += recall
    tFScore += fscore

    
    # Check second fold
    neigh = KNeighborsClassifier(n_neighbors=k, metric=metric)
    Xtrain = train2.drop('K', axis = 1)
    Ytrain = train2.K
    neigh.fit(Xtrain, Ytrain)
    
    Xtest = test2.drop('K', axis = 1)
    Ytest = test2.K
    YPred = neigh.predict(Xtest)

    if i == 0:
        ConfusionMatrixY = ConfusionMatrixY + Ytest.tolist()
        ConfusionMatrixYpred = ConfusionMatrixYpred + YPred.tolist()
    
    accuracy, precision, recall, fscore = countStats(Ytest, YPred)
    tAccuracy += accuracy
    tPrecision += precision
    tRecall += recall
    tFScore += fscore

    # Create and normalize confusion matrix
    cm = confusion_matrix(ConfusionMatrixY, ConfusionMatrixYpred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
  return tAccuracy/10, tPrecision/10, tRecall/10, tFScore/10, cm

def calculateStatsForNM(data, metric='euclidean', drawConfusionMatrix=False):
    tAccuracy = 0
    tPrecision = 0
    tRecall = 0
    tFScore = 0

    # Data used to draw confusion matrix will be gathered here
    ConfusionMatrixY = []
    ConfusionMatrixYpred = []
  
    for i in range(0, 5):
        kf = KFold(n_splits = 2, shuffle = True)
        result = next(kf.split(data), None)

        train1 = data.iloc[result[0]]
        test1 =  data.iloc[result[1]]
        train2 = data.iloc[result[1]]
        test2 =  data.iloc[result[0]]

        # Check first fold
        nm = NearestCentroid(metric=metric)
        Xtrain = train1.drop('K', axis = 1)
        Ytrain = train1.K
        nm.fit(Xtrain, Ytrain)

        Xtest = test1.drop('K', axis = 1)
        Ytest = test1.K
        YPred = nm.predict(Xtest)

        if i == 0:
            ConfusionMatrixY = Ytest.tolist()
            ConfusionMatrixYpred = YPred.tolist()

        accuracy, precision, recall, fscore = countStats(Ytest, YPred)
        tAccuracy += accuracy
        tPrecision += precision
        tRecall += recall
        tFScore += fscore

        # Check second fold
        nm = NearestCentroid(metric=metric)
        Xtrain = train2.drop('K', axis = 1)
        Ytrain = train2.K
        nm.fit(Xtrain, Ytrain)

        Xtest = test2.drop('K', axis = 1)
        Ytest = test2.K
        YPred = nm.predict(Xtest)

        if i == 0:
            ConfusionMatrixY = ConfusionMatrixY + Ytest.tolist()
            ConfusionMatrixYpred = ConfusionMatrixYpred + YPred.tolist()

        accuracy, precision, recall, fscore = countStats(Ytest, YPred)
        tAccuracy += accuracy
        tPrecision += precision
        tRecall += recall
        tFScore += fscore

        # Create and normalize confusion matrix
        cm = confusion_matrix(ConfusionMatrixY, ConfusionMatrixYpred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return tAccuracy/10, tPrecision/10, tRecall/10, tFScore/10, cm
