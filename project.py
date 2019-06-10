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

best_data_euc_norm = None
best_data_man_norm = None
best_data_euc_not_norm = None
best_data_man_not_norm = None

# Load Data (param whether data should be normalized)
for norm_type in ['Norm', 'NotNorm']:
    data = getDataSetLeukemia(True if norm_type == 'Norm' else False)
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

    print (f'Features Ranking {sortedFeatures}', file=open('results/features_ranking.txt', 'w'))

    # WYKRES
    sns.set()
    '''
    sns.set_palette(sns.color_palette("hls", len(sortedFeatures)))
    # sns.pairplot(data, vars=sortedFeatures, hue='K', diag_kind='kde', dropna=True, kind='scatter')
    sns.pairplot(data, vars=sortedFeatures, hue='K', diag_kind='hist', dropna=True, kind='scatter')
    # plt.show()
    plt.savefig('results/' + str(norm_type) + '/' + str(norm_type) + '_Feature_plot.png')
    '''

    # --------------------------------------------------------

    k_list = list(range(1, 50, 2))
    f_scores = []

    # RÓŻNE WARTOŚCI K - EUCLIDIAN
    for k in k_list:
        accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(data, k, 'euclidean')
        f_scores.append(fscore)
        
    sns.set_style("whitegrid")

    plt.figure()
    plt.title('The optimal number of neighbors(metric: euclidean)', fontweight='bold')
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('fscore',)
    plt.plot(k_list, f_scores)

    plt.savefig('results/' + str(norm_type) + '/' + str(norm_type) + '_Optimal_neighbors_euclidean.png')
    best_k_euc = k_list[f_scores.index(max(f_scores))]
    print(f'Optimal neighbors euclidean: {best_k_euc}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_K_euclidean.txt', 'w'))

    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(data, best_k_euc, 'manhattan')
    print(f'Stats {str(best_k_euc)}-NN: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_K_euclidean.txt', 'a'))


    Optimal_neighbors_data_euc = pd.DataFrame(list(zip(k_list, f_scores)), columns = ['Neighbor', 'fscore'])
    print(f'Data:\n {Optimal_neighbors_data_euc.to_string()}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_K_euclidean.txt', 'a'))

    # --------------------------------------------------------

    k_list = list(range(1, 50, 2))
    f_scores = []
    # RÓŻNE WARTOŚCI K - MANHATTAN
    for k in k_list:
        accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(data, k, 'manhattan')
        f_scores.append(fscore)


    plt.figure()
    plt.title('The optimal number of neighbors(metric: manhattan)', fontweight='bold')
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('fscore',)
    plt.plot(k_list, f_scores)

    plt.savefig('results/' + str(norm_type) + '/' + str(norm_type) + '_Optimal_neighbors_manhattan.png')
    best_k_man = k_list[f_scores.index(max(f_scores))]
    print(f'Optimal neighbors manhattan: {best_k_man}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_K_manhattan.txt', 'w'))

    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(data, best_k_man, 'manhattan')
    print(f'Stats {str(best_k_man)}-NN: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_K_manhattan.txt', 'a'))


    Optimal_neighbors_data_man = pd.DataFrame(list(zip(k_list, f_scores)), columns = ['Neighbor', 'fscore'])
    print(f'Data:\n {Optimal_neighbors_data_man.to_string()}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_K_manhattan.txt', 'a'))

    # --------------------------------------------------------
    feature_list = range(1, len(sortedFeatures) + 1)
    f_scores_euc = []
    f_scores_NM_euc = []
    sub_dates_euc = []

    # RÓŻNA LICZBA CECH DLA WYLICZONEGO BESTA - EUC
    for feature in feature_list:
        subData = None
        subData = data.copy()
        for dropedFeatureIndex in range (feature, len(sortedFeatures)):
            subData = subData.drop(sortedFeatures[dropedFeatureIndex], axis=1)
        sub_dates_euc.append(subData)

        # Print stats For NM
        accuracyNM, precisionNM, recallNM, fscoreNM, confusionMatrixNM = calculateStatsForNM(subData, 'euclidean', True)
        f_scores_NM_euc.append(fscoreNM)

        # Print stats for best_k_euc-NN
        accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(subData, best_k_euc, 'euclidean')
        f_scores_euc.append(fscore)

    plt.figure()
    plt.title('The optimal number of features(metric: euclidean)', fontweight='bold')
    plt.xlabel('Number of Features')
    plt.ylabel('fscore',)
    plt.plot(feature_list, f_scores_euc)
    plt.savefig('results/' + str(norm_type) + '/' + str(norm_type) + '_Optimal_features_euclidean.png')

    plt.figure()
    plt.title('The optimal number of features(metric: euclidean)', fontweight='bold')
    plt.xlabel('Number of Features')
    plt.ylabel('fscore',)
    plt.plot(feature_list, f_scores_NM_euc)
    plt.savefig('results/' + str(norm_type) + '/' + str(norm_type) + '_Optimal_featuresNM_euclidean.png')

    best_feat_euc = feature_list[f_scores_euc.index(max(f_scores_euc))]
    best_feat_NM_euc = feature_list[f_scores_NM_euc.index(max(f_scores_NM_euc))]

    print(f'Optimal features euclidean: {best_feat_euc} for best k: {best_k_euc}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_euclidean.txt', 'w'))
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(sub_dates_euc[best_feat_euc - 1], best_k_euc, 'euclidean')
    print(f'Stats {best_k_euc}-NN, Features {best_feat_euc}: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_euclidean.txt', 'w'))

    Optimal_features_data_euc = pd.DataFrame(list(zip(feature_list, f_scores_euc)), columns = ['Feature', 'fscore'])
    print(f'Data:\n {Optimal_features_data_euc.to_string()}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_euclidean.txt', 'a'))


    print(f'Optimal features euclidean: {best_feat_NM_euc} for NM', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_NM_euclidean.txt', 'w'))
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForNM(sub_dates_euc[best_feat_NM_euc - 1], 'euclidean', True)
    print(f'Stats NM, Features {best_feat_NM_euc}: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_NM_euclidean.txt', 'w'))

    Optimal_features_data_NM_euc = pd.DataFrame(list(zip(feature_list, f_scores_NM_euc)), columns = ['Feature', 'fscore'])
    print(f'Data:\n {Optimal_features_data_NM_euc.to_string()}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_NM_euclidean.txt', 'a'))

    # --------------------------------------------------------
    feature_list = range(1, len(sortedFeatures) + 1)
    f_scores_man = []
    f_scores_NM_man = []
    sub_dates_man = []

    # RÓŻNA LICZBA CECH DLA WYLICZONEGO BESTA - MAN
    for feature in feature_list:
        subData = None
        subData = data.copy()
        for dropedFeatureIndex in range (feature, len(sortedFeatures)):
            subData = subData.drop(sortedFeatures[dropedFeatureIndex], axis=1)
        sub_dates_man.append(subData)

        # Print stats For NM
        accuracyNM, precisionNM, recallNM, fscoreNM, confusionMatrixNM = calculateStatsForNM(subData, 'manhattan', True)
        f_scores_NM_man.append(fscoreNM)

        # Print stats for best_k_man-NN
        accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(subData, best_k_man, 'manhattan')
        f_scores_man.append(fscore)

    plt.figure()
    plt.title('The optimal number of features(metric: manhattan)', fontweight='bold')
    plt.xlabel('Number of Features')
    plt.ylabel('fscore',)
    plt.plot(feature_list, f_scores_man)
    plt.savefig('results/' + str(norm_type) + '/' + str(norm_type) + '_Optimal_features_manhattan.png')

    plt.figure()
    plt.title('The optimal number of features(metric: manhattan)', fontweight='bold')
    plt.xlabel('Number of Features')
    plt.ylabel('fscore',)
    plt.plot(feature_list, f_scores_NM_man)
    plt.savefig('results/' + str(norm_type) + '/' + str(norm_type) + '_Optimal_featuresNM_manhattan.png')

    best_feat_man = feature_list[f_scores_man.index(max(f_scores_man))]
    best_feat_NM_man = feature_list[f_scores_NM_man.index(max(f_scores_NM_man))]

    print(f'Optimal features manhattan: {best_feat_man} for best k: {best_k_man}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_manhattan.txt', 'w'))
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForKMeans(sub_dates_man[best_feat_man - 1], best_k_man, 'manhattan')
    print(f'Stats {best_k_man}-NN, Features {best_feat_man}: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_manhattan.txt', 'w'))

    Optimal_features_data_man = pd.DataFrame(list(zip(feature_list, f_scores_man)), columns = ['Feature', 'fscore'])
    print(f'Data:\n {Optimal_features_data_man.to_string()}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_manhattan.txt', 'a'))


    print(f'Optimal features manhattan: {best_feat_NM_man} for NM', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_NM_manhattan.txt', 'w'))
    accuracy, precision, recall, fscore, confusionMatrix = calculateStatsForNM(sub_dates_man[best_feat_NM_man - 1], 'manhattan', True)
    print(f'Stats NM, Features {best_feat_NM_man}: \n\t Accuracy: {accuracy} \n\t Precision: {precision} \n\t Recall: {recall} \n\t Fscore: {fscore} \n\t Confusion Matrix: \n {confusionMatrix}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_NM_manhattan.txt', 'w'))

    Optimal_features_data_NM_man = pd.DataFrame(list(zip(feature_list, f_scores_NM_man)), columns = ['Feature', 'fscore'])
    print(f'Data:\n {Optimal_features_data_NM_man.to_string()}', file=open('results/' + str(norm_type) + '/' + str(norm_type) + '_Diffrent_Features_NM_manhattan.txt', 'a'))
    
    '''
    print(type(confusionMatrix))
    print(sub_dates_man[best_feat_NM_man - 1])
    print(confusionMatrix.shape)

    df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJK"],
                  columns = [i for i in "ABCDEFGHIJK"])
    sn.heatmap(df_cm, annot=True)
    '''
    print(type(sub_dates_man))
    print(len(sub_dates_man))
    # print(sub_dates_man)
    print(len(sub_dates_man[0]))
    print(len(sub_dates_man[10]))
    