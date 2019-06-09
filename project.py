import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing
from stats import calculateStatsForKMeans, calculateStatsForNM
from dataLoad import getDataSetLeukemia, getDataSetIris
import seaborn as sns

# Load Data (param whether data should be normalized)
# data = getDataSetLeukemia(False)
data = getDataSetIris()
# K - class
data.K = data.K.astype(str)
data.K.replace('0.0', 'iris setosa')
X = data.drop('K', axis = 1)
y = data.K

'''
# DANE O INDEXIE
print(f'dtypes: {data.dtypes}')
print(f'y: {y}')
print(f'y[-1]: {y.iloc[-1]}')
print(f'type(y[-1]): {type(y.iloc[-1])}')
'''


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


# print(data)
# Wyniki - dane

'''
# INFO
print(data)
print(f'Zależność liniowa(powinna być równa ilości kolumn): {np.linalg.matrix_rank(X)}')
print(f'Kształt danych: {data.shape}')
print(f'Cechy kluczowe: {sortedFeatures}')
print(f'Długość cech kluczowych: {len(sortedFeatures)}')
print(f'Wymiar danyych: {data.ndim}')
print(f'Info: {data.info(verbose=True)}')
print(f'data.T: {data.T}')
'''




''' 
# PRINT DATAFRAME
print(data.to_string(index=False, header=False, columns=data.columns.tolist()[:-1]), file=open('dataTest.txt', 'w'))
print(data.to_string(), file=open('dataTest.txt', 'w'))
print(f'data.T.dot(data): {data.T.dot(data).to_string()}', file=open('dataTDot.txt', 'w'))
'''


# WYKRES

sns.set()
sns.set_palette(sns.color_palette("hls", len(sortedFeatures)))
# sns.pairplot(data, vars=sortedFeatures, hue='K', diag_kind='kde', dropna=True, kind='scatter')
sns.pairplot(data, vars=sortedFeatures, hue='K', diag_kind='hist', dropna=True, kind='scatter')
# plt.show()
plt.savefig('Feature_plot.png')

k_list = list(range(1, 50))
f_scores = []

# --------------------------------------------------------

# RÓŻNE WARTOŚCI K - EUCLIDIAN
for k in k_list:
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(data, k, 'euclidean')
    f_scores.append(fscore)
    # print(f'Stats {str(k)}-NN: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open(filename, 'a'))

sns.set_style("whitegrid")

plt.figure()
plt.title('The optimal number of neighbors(metric: euclidean)', fontweight='bold')
plt.xlabel('Number of Neighbors K')
plt.ylabel('fscore',)
plt.plot(k_list, f_scores)

plt.savefig('Optimal_neighbors_euclidean.png')
best_k_euc = k_list[f_scores.index(max(f_scores))]
print(f'Optimal neighbors euclidean: {best_k_euc}', file=open('Diffrent_K_euclidean.txt', 'w'))
Optimal_neighbors_data_euc = pd.DataFrame(list(zip(k_list, f_scores)), columns = ['Neighbor', 'fscore'])
print(f'Data:\n {Optimal_neighbors_data_euc.to_string()}', file=open('Diffrent_K_euclidean.txt', 'a'))

# --------------------------------------------------------

k_list = list(range(1, 50))
f_scores = []
# RÓŻNE WARTOŚCI K - MANHATTAN
for k in k_list:
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(data, k, 'manhattan')
    f_scores.append(fscore)
    # print(f'Stats {str(k)}-NN: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open(filename, 'a'))


plt.figure()
plt.title('The optimal number of neighbors(metric: manhattan)', fontweight='bold')
plt.xlabel('Number of Neighbors K')
plt.ylabel('fscore',)
plt.plot(k_list, f_scores)

plt.savefig('Optimal_neighbors_manhattan.png')
best_k_man = k_list[f_scores.index(max(f_scores))]
print(f'Optimal neighbors manhattan: {best_k_man}', file=open('Diffrent_K_manhattan.txt', 'w'))
Optimal_neighbors_data_man = pd.DataFrame(list(zip(k_list, f_scores)), columns = ['Neighbor', 'fscore'])
print(f'Data:\n {Optimal_neighbors_data_man.to_string()}', file=open('Diffrent_K_manhattan.txt', 'a'))

# --------------------------------------------------------

# FILENAME VARIABLE
filename = 'Features_euc.txt'

# Wyniki
print (f'Features Ranking {sortedFeatures}', file=open(filename, 'w'))
print (f'\nResults:', file=open(filename, 'a'))

feature_list = range(1, len(sortedFeatures) + 1)
f_scores_euc = []

# RÓŻNA LICZBA CECH DLA WYLICZONEGO BESTA - EUC
for feature in feature_list:
    subData = data.copy()
    for dropedFeatureIndex in range (feature, len(sortedFeatures)):
        subData = subData.drop(sortedFeatures[dropedFeatureIndex], axis=1)
    print(f'Stats for features: {feature}', file=open(filename, 'a'))
    # Print stats For NM
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForNM(subData, 'euclidean', True)
    print(f'Stats NM: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open(filename, 'a'))

    # Print stats for best_k_euc-NN
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(subData, best_k_euc, 'euclidean')
    f_scores_euc.append(fscore)
    print(f'Stats {best_k_euc}-NN: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open(filename, 'a'))

    print(f'------------------', file=open(filename, 'a'))

plt.figure()
plt.title('The optimal number of features(metric: euclidean)', fontweight='bold')
plt.xlabel('Number of Features')
plt.ylabel('fscore',)
plt.plot(feature_list, f_scores_euc)

plt.savefig('Optimal_features_euclidean.png')
best_feat_euc = feature_list[f_scores_euc.index(max(f_scores_euc))]
print(f'Optimal features euclidean: {best_feat_euc}', file=open('Diffrent_Features_euclidean.txt', 'w'))
Optimal_features_data_euc = pd.DataFrame(list(zip(feature_list, f_scores_euc)), columns = ['Feature', 'fscore'])
print(f'Data:\n {Optimal_features_data_euc.to_string()}', file=open('Diffrent_Features_euclidean.txt', 'a'))

# --------------------------------------------------------

# FILENAME VARIABLE
filename = 'Features_man.txt'

# Wyniki
print (f'Features Ranking {sortedFeatures}', file=open(filename, 'w'))
print (f'\nResults:', file=open(filename, 'a'))

feature_list = range(1, len(sortedFeatures) + 1)
f_scores_man = []

# RÓŻNA LICZBA CECH DLA WYLICZONEGO BESTA - MAN
for feature in feature_list:
    subData = data.copy()
    for dropedFeatureIndex in range (feature, len(sortedFeatures)):
        subData = subData.drop(sortedFeatures[dropedFeatureIndex], axis=1)
    print(f'Stats for features: {feature}', file=open(filename, 'a'))
    # Print stats For NM
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForNM(subData, 'manhattan', True)
    print(f'Stats NM: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open(filename, 'a'))

    # Print stats for best_k_man-NN
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(subData, best_k_man, 'manhattan')
    f_scores_man.append(fscore)
    print(f'Stats {best_k_man}-NN: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open(filename, 'a'))

    print(f'------------------', file=open(filename, 'a'))

plt.figure()
plt.title('The optimal number of features(metric: manhattan)', fontweight='bold')
plt.xlabel('Number of Features')
plt.ylabel('fscore',)
plt.plot(feature_list, f_scores_man)

plt.savefig('Optimal_features_manhattan.png')
best_feat_man = feature_list[f_scores_man.index(max(f_scores_man))]
print(f'Optimal features manhattan: {best_feat_man}', file=open('Diffrent_Features_manhattan.txt', 'w'))
Optimal_features_data_man = pd.DataFrame(list(zip(feature_list, f_scores_man)), columns = ['Feature', 'fscore'])
print(f'Data:\n {Optimal_features_data_man.to_string()}', file=open('Diffrent_Features_manhattan.txt', 'a'))