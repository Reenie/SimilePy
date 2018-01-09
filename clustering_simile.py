from sklearn import svm
from matplotlib import style
import VectorSpace_simile as vs
import main
style.use("ggplot")
#import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from collections import Counter

import numpy as np
import pandas as pd
#import nltk
import re
import os
import codecs
from sklearn import feature_extraction
#import mpld3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, MeanShift
from sklearn.datasets import make_blobs

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler



#classification and dimensionality reduction
class Clustering_simile:
    def __init__(self):
        s = Clustering_simile #this class
        start = time.time()
        #s.lda_plot_2d_3d('self', main.filenames)
        s.clustering(self, clusters = 12, filenames= main.filenames, size_figure = (8, 6), random_state = 170, max_iter = 1000)
        #s.DBScan(self, filenames=main.filenames)
        #s.lda_plot_2d_3d_perSimile('self', main.filenames[0])
        #s.kmeans_Clustering(self)
        end = time.time()
        print("\n" + str(round((end - start), 3)) + " sec")



    def clustering(self, clusters = 5, filenames= main.filenames, size_figure = (6, 6), random_state = 170, max_iter = 300):
        s = Clustering_simile
        clustering_results, X, y_pred, y_train, target_values, feature_names, \
        featureValues_frequency_perCluster, distinct_vectors_perCluster = \
            s.kMeans(self, clusters = clusters, filenames=filenames, size_figure=(6, 6),
                     random_state=random_state, max_iter=max_iter, tol=1e-6, n_init=14)
        print(feature_names, end='\n\n')

        s.instancesOfClusters(self, clustering_results, featureValues_frequency_perCluster, distinct_vectors_perCluster)
        s.clustering_plot(self, X, y_pred, y_train, target_values, clusters=clusters, size_figure=size_figure)



    def instancesOfClusters(self, clustering_results, featureValues_frequency_perCluster, distinct_vectors_perCluster):
        s = Clustering_simile
        instances_perCluster = []
        instances_perSimile_perCluster = {}
        count_instances_perSimile_perCluster = {}
        instances_perSimile = {}
        count_instaces_perCluster = {}
        for cluster in clustering_results:
            count = 0
            count_instances_perSimile = {}
            instances_of_cluster_perSimile = {}
            for instance in cluster[1]:
                count += 1
                ############
                #print(instance)
                #instances_of_cluster.append((instance[0], instance[1]))
                if instance[0] in count_instances_perSimile:
                    temp = instances_of_cluster_perSimile[instance[0]]
                    temp.append(instance[1])
                    del instances_of_cluster_perSimile[instance[0]]
                    instances_of_cluster_perSimile[instance[0]] = temp
                    count_instances_perSimile[instance[0]] += 1
                else:
                    #perCent = str(100*instance[1]/instances_perSimile[instance[0]])+'%'
                    instances_of_cluster_perSimile[instance[0]] = [instance[1]]
                    count_instances_perSimile[instance[0]] = 1

                if instance[0] in instances_perSimile:
                    instances_perSimile[instance[0]] += 1
                else:
                    instances_perSimile[instance[0]] = 1
            count_instaces_perCluster[cluster[0]] = count
            count_instances_perSimile_perCluster[cluster[0]] = count_instances_perSimile
            instances_perSimile_perCluster[cluster[0]] = instances_of_cluster_perSimile
            sorted(count_instances_perSimile_perCluster)
            sorted(instances_perSimile_perCluster)
            #instances_perCluster.append(instances_of_cluster)

        #########
        print('Instances per Cluster:  \nfile:(number of instances, percentage of file)', end='')
        for i in count_instances_perSimile_perCluster:
            print('\n\nCluster ' + str(i) + ": (" + str(count_instaces_perCluster[i]) + " instances)\n", end="")
            for t in count_instances_perSimile_perCluster[i]:
                perCent = str(round(100*count_instances_perSimile_perCluster[i][t]/instances_perSimile[t],1))+'%'
                print(str(t) + ":(" + str(count_instances_perSimile_perCluster[i][t]) + ", " + perCent + ")  ", end="")
            #print("")
            #s.p(count_instances_perSimile_perCluster[i])

        #print the frequency of feature values per cluster
        print('\n\n\n\nFrequency of values of the features per cluster:', end='')
        for cluster_frequencyfeatureValues in featureValues_frequency_perCluster:
            cluster_no = cluster_frequencyfeatureValues[0]
            print('\n\nCluster: ' + str(cluster_no) + ": (" + str(
            count_instaces_perCluster[cluster_no]) + " instances)", end='\n')
            for k in cluster_frequencyfeatureValues[1]:
                if k[1]>0:
                    perCent = round(k[1]/count_instaces_perCluster[cluster_no]*100, 1)
                    print('(' + str(k[0]) + ": " + str(k[1]) + ', ' + str(perCent) + '%) ', end="")

        # Distinct vectors of values per cluster:
        print('\n\n\n\nDistinct vectors of values per cluster:\n[vector of values]: occurrences within the cluster, percentage within the cluster', end='')
        for cluster_ in distinct_vectors_perCluster:
            #print(cluster_)
            cluster_no = cluster_[0]
            cluster_vectors = cluster_[1]
            print('\n\nCluster: ' + str(cluster_no) + ": (" + str(
                        count_instaces_perCluster[cluster_no]) + " instances)", end='\n')
            #print(cluster_vectors)
            for vector in cluster_vectors:
                perCent = round(vector[1] / count_instaces_perCluster[cluster_no] * 100, 1)
                print('(' + str(vector[0]) + ": " + str(vector[1]) + ', ' + str(perCent) + '%)  ', end="")


        #count = 0
        print('\n\n\nInstances per Cluster: \nfile[number of excel row, ...]')
        for i in instances_perSimile_perCluster:
            print('\nCluster ' + str(i) + ": (" + str(count_instaces_perCluster[i]) + " instances)\n", end="")
            for t in instances_perSimile_perCluster[i]:
                print('file ' + str(t) + ': ', end='')
                print(instances_perSimile_perCluster[i][t])
            #count += len(i)
            #s.p(instances_perSimile_perCluster[i])
        #s.p(count)



    def DBScan(self, filenames = main.filenames):
        s = Clustering_simile
        x_train, x_test, y_train, y_test, x_train_instances, x_test_instances, target_values, feature_names = s.readAndSplitData(
            self, 1, filenames)
        x_d2, x_d3 = s.PCA_(self, x_train)  # principal component analysis (PCA)
        X = x_d2
        # #############################################################################
        # Compute DBSCAN
        db = DBSCAN(eps=0.12, min_samples=10).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
        # #############################################################################
        # Plot result
        import matplotlib.pyplot as plt
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()




    def kMeans(self, clusters = 5, filenames = main.filenames, size_figure = (6, 6), random_state = 170, max_iter = 900, tol = 1e-5, n_init=10):
        s = Clustering_simile
        x_train, x_test, y_train, y_test, x_train_instances, x_test_instances, target_values, feature_names = s.readAndSplitData(self, 1, filenames)
        x_d2 = s.PCA_(self, x_train)  # principal component analysis (PCA)
        X = x_d2
        y_pred = KMeans(n_clusters= clusters, tol=tol, random_state=random_state, max_iter=max_iter, n_init=10).fit_predict(X) #y_pred --> the number of cluster for each instance
        cluster_instances = {}
        for key, j, k in zip(y_pred, x_train_instances, x_train):
            key = key+1  #nombero fo cluster from 1, 2, ...
            # temp = cluster_instances.get(key=key, default=None) #returns value or default if key not in dictionary
            if key in cluster_instances:
                temp = cluster_instances[key]
                kk = [int(i) for i in k.tolist()]  # convert k from array to list and the elements from float to int
                temp.append((int(j[0]), int(j[1]), kk))
                del cluster_instances[key]
                cluster_instances[key] = temp
            else:
                kk = [int(i) for i in k.tolist()]  # convert k from array to list and the elements from float to int
                cluster_instances[key] = [(int(j[0]), int(j[1]), kk)]
        clustering_results = []
        for i in cluster_instances:
            instances = cluster_instances[i]
            instances.sort(key=lambda tup: (tup[0], tup[1]))
            clustering_results.append((i, instances))
        #clustering_results --> [(cluster, [(simile, xlsx_row, vector_of_row), ...all instances of the cluster...)], ....clusters...])]
        featureValues_frequency_perCluster,  distinct_vectors_perCluster= s.featuresOfClusters(self,
                                                             cluster_results=clustering_results,
                                                             feature_names=feature_names)
        return clustering_results, X, y_pred, y_train, target_values, feature_names, \
               featureValues_frequency_perCluster, distinct_vectors_perCluster


    #it returns the frequency of feature values per clauster
    def featuresOfClusters(self, cluster_results = [], feature_names = []):
        s = Clustering_simile
        featureValues_frequency_perCluster = []
        size = len(feature_names)
        feature_names = feature_names[3:]
        distinct_vectors_perCluster = []
        for cluster in cluster_results:
            cluster_no = cluster[0]
            cluster_instances = cluster[1]
            feature_frequences = [0] * size
            ##############
            #print(cluster_instances)
            #
            #frequency_of_distinctVectors = []
            vectorSpace = []
            for instance in cluster_instances:
                feature_index = -1
                for f_value, f_name in zip(instance[2], feature_names):
                            feature_index += 1
                            ###############
                            #print(str(f_name) + ": " + str(f_value))
                            if f_value == 1: #or feature!= 'nodet' or feature != 0:
                                feature_frequences[feature_index] += 1

                vectorSpace.append(instance[2])

            #distinct_vectors
            #distinct_vectors = [list(x) for x in set(tuple(x) for x in vectorSpace)]
            distinct_list = Counter([tuple(x) for x in vectorSpace])#.most_common(1)[0]  # Note 1
            keys = distinct_list.keys()
            distinct_vectors = []
            for k in keys:
                distinct_vectors.append((k, distinct_list[k]))
            distinct_vectors.sort(key=lambda tup: (-tup[1]))
            #print(distinct_vectors)
            distinct_values = []
            for vec in distinct_vectors:
                index = -1
                values = []
                for v in vec[0]:
                    index += 1
                    if v>0:
                        values.append(feature_names[index])
                distinct_values.append((values, vec[1]))
            distinct_vectors_perCluster.append((cluster_no, distinct_values))





            featureValuesFrequency = [] #[(value_name, frequency), ...]
            for value, freq in zip(feature_names, feature_frequences):
                featureValuesFrequency.append((value, freq))
            featureValuesFrequency.sort(key=lambda tup: (-tup[1]))
            featureValues_frequency_perCluster.append((cluster_no, featureValuesFrequency))
            #print(cluster_no)
        #for i in distinct_vectors_perCluster:
        #    print(i, end='\n\n')
        return featureValues_frequency_perCluster, distinct_vectors_perCluster





    def hasKey(self, hashTable, key):
        for k in hashTable.keys():
            if k == key:
                return 1
        return 0




    def clustering_plot(self, X, y_pred, y_train, target_values,  clusters, size_figure = (6, 6)):
        markers = ['o', '^', 'D', '>', '*', 'p', 'P', '1', 'X', 's',
                   'v', '8', 'h', '>', '|', '_', ',', '4', '2', '.']
        labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
        colors = ['blue', 'purple',  'green', 'red', 'black',  'orange', 'brown',     'cyan',  'magenta', 'yellow',   'turquoise',
                  '#029386', '#9a0eea', '#00035b', '#d1b2bf', '#1e1e6c', '#ff81c0', '#650021', '#c11afd', '#610023','#033500']

        plt.figure(1, figsize=size_figure)
        for i in range(clusters):
            #cluster_instances.append((i, x_train_instances[y_pred==i][0], x_train_instances[y_pred==i][1], x_train[y_pred==i]))
            #s.p(i+1)
            plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], c=colors[i], label = ('Cluster ' + str(i+1)))
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title("Clustering")

        plt.figure(2, figsize=size_figure)
        target_values = [int(i) for i in target_values]
        target_values.sort()
        for i in target_values:
            i = float(i)
            plt.scatter(X[y_train==i, 0], X[y_train==i, 1], color=colors[int(i)-1], marker=markers[int(i)-1], label = labels[int(i)-1])
        plt.title("Instances of 20 similes")
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.show()






    def readAndSplitData(self, training_fraction, filenames):
        s = Clustering_simile
        feature_names, vector_space = vs.VectorSpace_simile.numericalVectorSpace("self", filenames)
        np.random.shuffle(vector_space) #Reordering randomly the vector_space content
        x = np.array(vector_space)
        target_values = []
        # for y_target
        for tn in x[0:, 0]:
            if not (tn in target_values):
                target_values.append(tn)
        #print(feature_names)
        l = len(x)
        training_len = int(l * training_fraction)
        x_train = x[:training_len, 3:].astype(float)
        x_test = x[training_len:, 3:].astype(float)
        y_train = x[:training_len, 0].astype(float)  # target
        y_test = x[training_len:, 0].astype(float)  # target
        x_train_instances = x[:training_len, :2]
        x_test_instances = x[training_len:, :2]
        # print(x_train_instances)
        # target_values = target_values.astype(float)
        return x_train, x_test, y_train, y_test, x_train_instances, x_test_instances, target_values, feature_names








    def lda_plot_2d_3d(self, filenames = main.filenames):
        s = Clustering_simile  # this class
        figure_number = 0
        for i in range(1):
            figure_number += 2
            x_train, x_test, y_train, y_test, x_train_instances, x_test_instances, target_values, featureNames = s.readAndSplitData('self', 1, filenames)

            #x_d2, x_d3 = s.LSA(self, x_train)           #Latent Semantic Analysis
            #x_d2, x_d3 = s.LDA(self, x_train, y_train)  #Linear Discriminant Analysis
            #x_d2, x_d3 = s.NMF_(self, x_train, y_train) # Non-Negative Matrix Factorization
            x_d2, x_d3 = s.PCA_(self, x_train)           #principal component analysis (PCA)
            #x_d2, x_d3 = c.KPCA(self, x_train)           # Kernel principal component analysis (KPCA)
            colors = ['magenta', 'turquoise', 'brown', 'red', 'black', 'blue', 'cyan', 'green', 'orange', 'yellow',
                      '#029386', '#9a0eea', '#00035b' , '#d1b2bf', '#7e1e9c', '#ff81c0', '#650021', '#c11afd', '#610023', '#033500' ]
                      #'teal', 'pink', 'purple', 'grey', 'violet', 'dark blue', 'tan', 'forest green', 'olive', '#01153e']
            #colors = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19' ]
            markers = ['o', '^', 'D', '>', '*', 'p', 'P', '1', 'X', 's',
                       'v', '8', 'h', '>', '|', '_', ',', '4', '2', '.']
            labels = ["1_asp", "2_sto", "3_ap_p", "4_ap_x", "5_ela", "6_kok", "7_opl", "8_mal",
                      "9_ger", "10_pis", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
            #clustering.plot_2d_3d("self", i, colors, markers, x_d2, y_train, target_values)
            fig = plt.figure(figure_number-1)
            ax1 = fig.add_subplot(111)
            #print(target_values)
            for color, i, m in zip(colors, target_values, markers):
                ax1.scatter(x_d2[y_train == i, 0], x_d2[y_train == i, 1], alpha=.8, color=color, marker=m, label=labels[int(i)-1])
                # plt.scatter(x_d2[y_train == i, 0], x_d2[y_train == i, 1], alpha=.8, color=color, marker=m, label=int(i))
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('SIMILE (2D)')
            #plt.show()
            fig2 = plt.figure(figure_number)
            ax2 = fig2.add_subplot(111, projection='3d')
            for color, i, m in zip(colors, target_values, markers):
                ax2.scatter(x_d3[y_train == i, 0], x_d3[y_train == i, 1], x_d3[y_train == i, 2], alpha=.8, c=color,
                           marker=m, label=labels[int(i)-1])
                plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('SIMILE (3D)')
            plt.show()




    def lda_plot_2d_3d_perSimile(self, filename=main.filenames[1]):
            s = Clustering_simile  # this class
            figure_number = 0
            for i in range(1):
                figure_number += 2
                x_train, x_test, y_train, y_test, x_train_instances, x_test_instances, target_values, featureNames = s.readAndSplitData('self', 1, [filename])
                #x_d2, x_d3 = s.LSA(self, x_train)           #Latent Semantic Analysis
                #x_d2, x_d3 = s.LDA(self, x_train, y_train)  # Linear Discriminant Analysis
                #x_d2, x_d3 = s.NMF_(self, x_train, y_train) # Non-Negative Matrix Factorization
                x_d2, x_d3 = s.PCA_(self, x_train)           #principal component analysis (PCA)
                #x_d2, x_d3 = s.KPCA(self, x_train)           # Kernel principal component analysis (KPCA)
                color = 'magenta'
                # 'teal', 'pink', 'purple', 'grey', 'violet', 'dark blue', 'tan', 'forest green', 'olive', '#01153e']
                # colors = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19' ]
                marker = 'o'
                label = filename
                # clustering.plot_2d_3d("self", i, colors, markers, x_d2, y_train, target_values)
                fig = plt.figure(figure_number - 1)
                ax1 = fig.add_subplot(111)
                # print(target_values)
                #for color, i, m in zip(colors, target_values, markers):
                ax1.scatter(x_d2[y_train == i, 0], x_d2[y_train == i, 1], alpha=.8, color=color, marker=marker, label=label)
                plt.legend(loc='best', shadow=False, scatterpoints=1)
                plt.title('SIMILE (2D)')
                # plt.show()
                fig2 = plt.figure(figure_number)
                ax2 = fig2.add_subplot(111, projection='3d')
                #for color, i, m in zip(colors, target_values, markers):
                ax2.scatter(x_d3[y_train == i, 0], x_d3[y_train == i, 1], x_d3[y_train == i, 2], alpha=.8, color=color, marker=marker, label=label)
                plt.legend(loc='best', shadow=False, scatterpoints=1)
                plt.title('SIMILE (3D)')
                plt.show()






    #  Kernel Principal component analysis (IPCA)
    def KPCA(self, x_train):
            m_d2 = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=True, gamma=10)
            x_d2 = m_d2.fit_transform(x_train)
            x_d2 = m_d2.inverse_transform(x_d2)
            m_d3 = KernelPCA(n_components=3, kernel="rbf", fit_inverse_transform=True, gamma=10)
            x_d3 = m_d3.fit_transform(x_train)
            x_d3 = m_d3.inverse_transform(x_d3)
            return x_d2, x_d3

    # Principal component analysis (PCA)
    def PCA_(self, x_train):
        m_d2 = PCA(n_components=2, svd_solver='randomized')
        x_d2 = m_d2.fit_transform(x_train)
        #m_d3 = PCA(n_components=3, svd_solver='randomized')
        #x_d3 = m_d3.fit_transform(x_train)
        return x_d2 #x_d3


    #Non-Negative Matrix Factorization
    def NMF_(self, x_train, y_train):
        m_d2 = NMF(n_components=2, init='random', random_state=0)
        x_d2 = m_d2.fit(x_train, y_train).transform(x_train)
        m_d3 = NMF(n_components=3, init='random', random_state=0)
        x_d3 = m_d3.fit(x_train, y_train).transform(x_train)
        return x_d2, x_d3


    #Latent Semantic Analysis
    def LSA(self, x_train):
        svd = TruncatedSVD(n_components=2)
        normalizer = Normalizer(copy=False)
        m_d2 = make_pipeline(svd, normalizer)
        x_d2 = m_d2.fit_transform(x_train)
        svd = TruncatedSVD(n_components=3)
        m_d3 = make_pipeline(svd, normalizer)
        x_d3 = m_d3.fit_transform(x_train)
        return x_d2, x_d3

    #Linear Discriminant Analysis
    def LDA(self, x_train, y_train):
        lda_d2 = LinearDiscriminantAnalysis(solver='svd', n_components=2)
        x_d2 = lda_d2.fit(x_train, y_train).transform(x_train)
        lda_d3 = LinearDiscriminantAnalysis(solver='svd', n_components=3)
        x_d3 = lda_d3.fit(x_train, y_train).transform(x_train)
        return x_d2, x_d3






    def readAndSplitKFoldsData(self, folds=10, filenames=[]):
        feature_names, vector_space = vs.VectorSpace_simile.numericalVectorSpace("self", filenames)
        x = np.array(vector_space)
        #x = vector_space[1:, 0:]
        np.random.shuffle(x)
        x = x.astype(float)
        #print(x[0:, 0])
        target_values = []
        #for y_target
        for tn in x[0:, 0]:
            if not (tn in target_values):
                target_values.append(tn)
        #print(target_values)
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []
        kf = KFold(n_splits=folds)
        for train, test in kf.split(x): #train and test instances
            x_t = []
            y_t = []
            for i in train:
                x_t.append(x[i, 1:])
                y_t.append(x[i, 0])
            x_train_list.append(x_t)
            y_train_list.append(y_t)  #target
            x_t = []
            y_t = []
            for i in test:
                x_t.append(x[i, 1:])
                y_t.append(x[i, 0])
            x_test_list.append(x_t)
            y_test_list.append(y_t)  #target
        return x_train_list, x_test_list, y_train_list, y_test_list, target_values, feature_names

    def p(text):
        print(text)




Clustering_simile()
