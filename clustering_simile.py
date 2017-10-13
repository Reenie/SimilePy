from sklearn import svm
from matplotlib import style
import VectorSpace_simile as vs
import main
style.use("ggplot")
import random
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


#classification and dimensionality reduction
class Clustering_simile:
    def __init__(self):
        s = Clustering_simile #this class
        start = time.time()
        s.lda_plot_2d_3d('self')
        #s.kmeans_Clustering(self)
        end = time.time()
        print("\n" + str(round((end - start), 3)) + " sec")




    def kmeans_Clustering(self):
        s = Clustering_simile
        x_train, x_test, y_train, y_test, target_values, feature_names = s.readAndSplitData(self, 1)
        X = x_train[:, 3:]
        n_samples = len(X)
        print("n_samples: " + str(n_samples))
        random_state = 170
        x_d2, x_d3 = s.LDA(self, X, y_train)
        # Incorrect number of clusters
        y_pred = KMeans(n_clusters=10, random_state=random_state).fit_predict(x_d3)
        plt.subplot(111)
        plt.scatter(x_d3[:, 0], x_d3[:, 1], x_d3[:, 2], c=y_pred)

        plt.show()


    def lda_plot_2d_3d(self):
        s = Clustering_simile  # this class
        figure_number = 0
        for i in range(3):
            figure_number +=2
            x_train, x_test, y_train, y_test, target_values, featureNames = s.readAndSplitData('self', 1)
            #x_d2, x_d3 = c.LSA(self, x_train)           #Latent Semantic Analysis
            x_d2, x_d3 = s.LDA(self, x_train, y_train)  #Linear Discriminant Analysis
            #x_d2, x_d3 = c.NMF_(self, x_train, y_train) # Non-Negative Matrix Factorization
            #x_d2, x_d3 = c.PCA_(self, x_train)           #principal component analysis (PCA)
            #x_d2, x_d3 = c.KPCA(self, x_train)           # Kernel principal component analysis (KPCA)
            colors = ['magenta', 'turquoise', 'brown', 'red', 'black', 'blue', 'cyan', 'green', 'orange', 'yellow',
                      '#029386', '#9a0eea', '#00035b' , '#d1b26f', '#7e1e9c', '#ff81c0', '#650021', '#cea2fd', '#610023', '#033500' ]
                      #'teal', 'pink', 'purple', 'grey', 'violet', 'dark blue', 'tan', 'forest green', 'olive', '#01153e']
            #colors = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19' ]
            markers = ['o', '^', 'D', '>', '*', 'p', 'P', '1', 'X', 's',
                       'o', '^', 'D', '>', '*', 'p', 'P', '1', 'X', 's']
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
        m_d3 = PCA(n_components=3, svd_solver='randomized')
        x_d3 = m_d3.fit_transform(x_train)
        return x_d2, x_d3


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






    def readAndSplitKFoldsData(self, folds=10):
        feature_names, vector_space = vs.VectorSpace_simile.numericalVectorSpace("self", main.filenames)
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
        for train, test in kf.split(x):
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



    def readAndSplitData(self, training_fraction):
        feature_names, vector_space = vs.VectorSpace_simile.numericalVectorSpace("self", main.filenames)
        x = np.array(vector_space)
        #x = vector_space[0:, 0:]
        x = x.astype(float)
        target_values = []
        #for y_target
        for tn in x[0:, 0]:
            if not (tn in target_values):
                target_values.append(tn)
        #print(feature_names)
        random.shuffle(x)
        l = len(x)
        training_len = int(l*training_fraction)
        x_train = x[:training_len, 3:]
        x_test = x[training_len:, 3:]
        y_train = x[:training_len, 0]  #target
        y_test = x[training_len:, 0]   #target
        return x_train, x_test, y_train, y_test, target_values, feature_names



Clustering_simile()
