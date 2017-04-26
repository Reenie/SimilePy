from sklearn import svm

from matplotlib import style
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

import VectorSpace_v3 as vs3
import main
from Classifiers import Classifiers

style.use("ggplot")
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
import pandas
import numpy
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import chi2

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing


#classification and dimensionality reduction
class Classification_simile:
    def __init__(self):
        s = Classification_simile #this class
        start = time.time()
        #s.classifier_evaluation('self', classifier = 2)
        #s.evaluateAllClassifiers(self, numOfClassifiers=7)
        s.UFS_featureSelection(self, 25)
        #s.RFE_featureSelection(self)
        #s.TBFS_featureSelection(self)
        #s.lda_plot_2d_3d('self')
        end = time.time()
        print("\n" + str(round((end - start), 3)) + " sec")



    #Univariate feature selection
    # Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
    def UFS_featureSelection(self, kbest=1000):
        s = Classification_simile  # this class
        # load data
        x_train, x_test, y_train, y_test, target_values, feature_names = s.readAndSplitData('self', 1)

        names = feature_names[3:]
        #print(names)
        X = x_train[0:, 2:]
        Y = y_train
        # feature extraction
        test = SelectKBest(score_func=chi2, k='all')
        #test = SelectKBest(score_func=f_classif, k='all')
        #test = SelectKBest(score_func=mutual_info_classif, k='all')
        fit = test.fit(X, Y)
        # summarize scores
        numpy.set_printoptions(precision=3)
        #print(fit.scores_)
        features = fit.transform(X)
        # summarize selected features
        #print(features[0:, :])
        #print(len(fit.scores_))
        #print(len(feature_names))
        score_attr_tupple = []
        for n, sc, p in zip(names, fit.scores_, fit.pvalues_):
            # print(str(a) + " " + b )
            if np.math.isnan(sc):
                sc = 0.0
            score_attr_tupple.append((n, sc, p))
        score_attr_tupple.sort(key=lambda tup: tup[1], reverse=True)
        # str_for_printing = "Univariate feature selection (best " + str(kbest) + " features):\n"
        count_best = 0
        print("Univariate feature selection (best " + str(kbest) + " features):\n")
        print("%-14s%-14s" % ("Attribute", "Score"))
        for t in score_attr_tupple:
            # print("%-14s%-14s" % (str(a[2]), str(round(a[0], 3))))
            print(str(t[0]) + ", " + str(round(t[1], 3)))
            # print(str(t[0]) +", "  + str(round(t[1], 3)) + ", "+ str(round(t[2],15)) + ", " +  str(t[1]/t[2]))
            # str_for_printing += "(" + str(a[2]) + ", " + str(a[1]) + ", " + str(round(a[0], 3)) + "),  "
            count_best += 1
            if kbest == count_best:
                break
                # print(str_for_printing)
                #    print(str(round(a[0],3)) + " " + str(a[1]))



    # Recursive Feature Elimination works by recursively removing attributes and building a model on those attributes that remain.
    def RFE_featureSelection(self):
        s = Classification_simile  # this class
        # load data
        x_train, x_test, y_train, y_test, target_values, feature_names = s.readAndSplitData('self', 1)
        names = feature_names[3:]
        print(names)
        X = x_train[0:, 2:]
        Y = y_train
        # feature extraction
        model = svm.LinearSVC()
        rfe = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
                      scoring='accuracy')
        fit = rfe.fit(X, Y)
        features = fit.n_features_
        print(features)
        print(len(fit.ranking_))
        print(len(feature_names))
        score_attr_tupple = []
        for a, b in zip(fit.ranking_, names):
            # print(str(a) + " " + b )
            if np.math.isnan(a):
                a = 0.0
            score_attr_tupple.append((a, b))
        score_attr_tupple.sort(key=lambda tup: tup[0], reverse=False)
        for a in score_attr_tupple:
            print(str(round(a[0], 3)) + " " + str(a[1]))



    #Tree-based feature selection
    #Tree - based estimators can be used to compute feature importances, which in turn can be used to discard irrelevant features
    def TBFS_featureSelection(self):
        s = Classification_simile  # this class
        # load data
        x_train, x_test, y_train, y_test, target_values, feature_names = s.readAndSplitData('self', 1)
        names = feature_names[3:]
        print(names)
        X = x_train[0:, 2:]
        Y = y_train
        # feature extraction
        model = ExtraTreesClassifier()
        fit = model.fit(X, Y)
        features = fit.n_features_
        print(features)
        print(len(fit.feature_importances_))
        print(len(names))
        score_attr_tupple = []
        for a, b in zip(fit.feature_importances_, names):
            # print(str(a) + " " + b )
            if np.math.isnan(a):
                a = 0.0
            score_attr_tupple.append((a, b))
        score_attr_tupple.sort(key=lambda tup: tup[0], reverse=True)
        for a in score_attr_tupple:
            print(str(round(a[0], 3)) + " " + str(a[1]))



        # LDA - Linear Discriminant Analysis
    def classifier_evaluation(self, classifier=1):
        s = Classification_simile  # this class
        x_train_list, x_test_list, y_train_list, y_test_list, target_values = \
            s.readAndSplitKFoldsData('self', 2)
        accuracy = []
        precision_list = []
        recall_list = []
        fscore_list = []
        suport_list = []
        labels_list = []
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            #print(x_test)
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            #x_train = preprocessing.normalize(x_train, norm='l2')
            #x_test =preprocessing.normalize(x_test, norm='l2')
            #x_train = preprocessing.scale(x_train)
            #x_test = preprocessing.scale(x_test)
            #print(x_test)
            print(x_train[0:, 2:].shape)
            print(y_train.shape)
            print(x_test[0:, 2:].shape)
            y_pred = Classifiers.run_classifier(self, x_train[0:, 2:], y_train, x_test[0:, 2:], classifier=classifier)
            t = 0
            f = 0
            for y, y_t in zip(y_pred, y_test):
                if (y == y_t):
                    t += 1
                    # print(str(y) + " _ " + str(y_t))
                else:
                    f += 1
                    # print(str(y) + " _ " + str(y_t) + ' Error')
            print(classification_report(y_test, y_pred))
            print("True: " + str(t) + " False: " + str(f))
            print("accuracy: " + str(round((t / (t + f)), 3)) + "\n")
            accuracy.append(t / (t + f))
            labels = unique_labels(y_test, y_pred)
            labels_list.append(labels)
            #print(labels)
            pr, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)
            precision_list.append(pr)
            recall_list.append(rec)
            fscore_list.append(fs)
            suport_list.append(sup)
        labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support = \
            s.meanOfLists(self, labels_list, precision_list, recall_list, fscore_list, suport_list)
        labels = ["1_aspr", "2_stol", "3_apal_p", "4_apal_x", "5_elafr", "6_kokkin", "7_oplism", "8_malak", "9_geros", "10_pist" ]
        print('%-14s%-14s%-14s%-14s%-14s' % ("Simile", "Precision", "Recall", "F1-score", "Support"))
        #labels = labels_list[0]
        for l, p, r, f, s in zip(labels, precision, recall, fscore, support):
            tuple = (l, round(p, 3), round(r, 3), round(f, 3), int(round(s)))
            print('%-14s%-14s%-14s%-14s%-14s' % tuple)
        tuple = ('\nAvg/Total', round(avg_precision,3),round(avg_recall,3), round(avg_fscore,3), int(round(total_support)))
        print('%-14s%-14s%-14s%-14s%-14s' % tuple)
        print("avg accuracy: " + str(round(np.mean(accuracy),3)))
        return avg_fscore



    def evaluateAllClassifiers(self, numOfClassifiers=8):
        s = Classification_simile
        c = Classifiers
        classifier_prec_rec_fScor = []
        for i in range(numOfClassifiers):
            pr, rec, f1 = s.classifier_evaluation_withoutPrint("self", classifier=(i+1))
            classifier_prec_rec_fScor.append([i+1, pr, rec, f1])
        print('%-14s%-14s%-14s%-14s' % ("Classifier", "Precision", "Recall", "F1-Score"))
        for v in classifier_prec_rec_fScor:
            print('%-14s%-14s%-14s%-14s' % (str(c.classifier_names[v[0]]), str(round(v[1],3)), str(round(v[2],3)), str(round(v[3],3))))



    def classifier_evaluation_withoutPrint(self, classifier=1):
        s = Classification_simile  # this class
        x_train_list, x_test_list, y_train_list, y_test_list, target_values = \
            s.readAndSplitKFoldsData('self', 10)
        accuracy = []
        precision_list = []
        recall_list = []
        fscore_list = []
        suport_list = []
        labels_list = []
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            #preprocessing.normalize(x_train, norm='l2')
            #preprocessing.normalize(x_test, norm='l2')
            y_pred = Classifiers.run_classifier(self, x_train[0:, 2:], y_train, x_test[0:, 2:], classifier=classifier)
            t = 0
            f = 0
            for y, y_t in zip(y_pred, y_test):
                if (y == y_t):
                    t += 1
                else:
                    f += 1
            accuracy.append(t / (t + f))
            labels = unique_labels(y_test, y_pred)
            labels_list.append(labels)
            # print(labels)
            pr, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)
            precision_list.append(pr)
            recall_list.append(rec)
            fscore_list.append(fs)
            suport_list.append(sup)
        labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support = \
            s.meanOfLists(self, labels_list, precision_list, recall_list, fscore_list, suport_list)
        return avg_precision, avg_recall, avg_fscore



    #it returns the arrays of the mean value of evaluation metrics and the average of each array
    def meanOfLists(self, labels_list, precision_list, recall_list, fscore_list, suport_list):
        max_size = 0
        for l in labels_list:
            size = len(l)
            if size > max_size:
                max_size = size
        precision = []
        recall = []
        fscore = []
        support = []
        count = []
        for i in range(max_size):
            precision.append(0.0)
            recall.append(0.0)
            fscore.append(0.0)
            support.append(0.0)
            count.append(0)
        for labels, pr, re, fs, su in zip(labels_list, precision_list, recall_list, fscore_list, suport_list):
            for l, p, r, f, s in zip(labels, pr, re, fs, su):
                i = int(l)-1
                count[i] += 1
                precision[i] += p*s
                recall[i] += r*s
                fscore[i] += f*s
                support[i] += s
        #print(count)
        index = -1
        for p, r, f, s, c in zip(precision, recall, fscore, support, count):
            index += 1
            precision[index] = p/s
            recall[index] = r/s
            fscore[index] = f/s
            support[index] = s/c
                #precision[i] = (precision[i] * (count[i] - 1) + p) / count[i]
                #recall[i] = (recall[i] * (count[i] - 1) + r) / count[i]
                #fscore[i] = (fscore[i] * (count[i] - 1) + f) / count[i]
                #support[i] = (support[i] * (count[i] - 1) + s) / count[i]
        labels = []
        for i in range(max_size):
            labels.append(i+1)
        avg_precision = 0.0
        avg_recall = 0.0
        avg_fscore = 0.0
        total_support = 0.0
        for p, r, f, s, in zip(precision, recall, fscore, support):
            avg_precision += p*s
            avg_recall += r*s
            avg_fscore += f*s
            total_support += s
        avg_precision = avg_precision/total_support
        avg_recall = avg_recall/total_support
        avg_fscore = avg_fscore/total_support
        return labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support



    def lda_plot_2d_3d(self):
        s = Classification_simile  # this class
        figure_number = 0
        for i in range(3):
            figure_number +=2
            x_train, x_test, y_train, y_test, target_values = s.readAndSplitData('self', 1)
            #x_d2, x_d3 = c.LSA(self, x_train)           #Latent Semantic Analysis
            x_d2, x_d3 = s.LDA(self, x_train, y_train)  #Linear Discriminant Analysis
            #x_d2, x_d3 = c.NMF_(self, x_train, y_train) # Non-Negative Matrix Factorization
            #x_d2, x_d3 = c.PCA_(self, x_train)           #principal component analysis (PCA)
            #x_d2, x_d3 = c.KPCA(self, x_train)           # Kernel principal component analysis (KPCA)
            colors = ['magenta', 'turquoise', 'brown',
                  'red', 'black', 'blue',
                  'cyan', 'green', 'orange',
                  'yellow']
            #colors = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
            markers = ['o', '^', 'D', '>', '*', 'p', 'P', '1', 'X', 's']
            labels = ["1_asp", "2_sto", "3_ap_p", "4_ap_x", "5_ela", "6_kok", "7_opl", "8_mal",
                      "9_ger", "10_pis"]
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


    #LDA - Linear Discriminant Analysis
    def lda_plot(self):
        s = Classification_simile  # this class
        x_train, x_test, y_train, y_test, target_values = s.readAndSplitData('self', 1)
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_r2 = lda.fit(x_train, y_train).transform(x_train)
        colors = ['magenta', 'turquoise', 'brown',
                  'red', 'black', 'blue',
                  'pink', 'green', 'orange',
                  'yellow']
        for color, i in zip(colors, target_values):
            plt.scatter(X_r2[y_train == i, 0], X_r2[y_train == i, 1], alpha=.8, color=color, label=int(i))
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('LDA of simile dataset')
        plt.show()







    def plot_2d_3d(self, figure_number,  colors, markers, x_d2, y_train, target_values):
        fig = plt.figure(figure_number)
        ax = fig.add_subplot(111)
        for color, i, m in zip(colors, target_values, markers):
            ax.scatter(x_d2[y_train == i, 0], x_d2[y_train == i, 1], alpha=.8, color=color, marker=m, label=int(i))
            #plt.scatter(x_d2[y_train == i, 0], x_d2[y_train == i, 1], alpha=.8, color=color, marker=m, label=int(i))
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('LDA of simile dataset (2D)')
        #plt.show()
        fig = plt.figure(figure_number+1)
        ax = fig.add_subplot(111, projection='3d')
        for color, i, m in zip(colors, target_values, markers):
            ax.scatter(x_d2[y_train == i, 0], x_d2[y_train == i, 1], x_d2[y_train == i, 2], alpha=.8, c=color, marker=m, label=int(i))
        plt.title('LDA of simile dataset (3D)')



    #LDA - Linear Discriminant Analysis
    def lda_plot_3d(self):
        s = Classification_simile  # this class
        x_train, x_test, y_train, y_test, target_values = s.readAndSplitData('self', 1)
        lda = LinearDiscriminantAnalysis(solver='svd', n_components=3)
        X_r2 = lda.fit(x_train, y_train).transform(x_train)
        colors = ['magenta', 'turquoise', 'brown',
                  'red', 'black', 'blue',
                  'pink', 'green', 'orange',
                  'yellow']
        markers = ['o', '^', '<', '>', 's', 'p', 'h', '+', 'x', '|']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for color, i, m in zip(colors, target_values, markers):
            ax.scatter(X_r2[y_train == i, 0], X_r2[y_train == i, 1], X_r2[y_train == i, 2], alpha=.8, c=color, marker=m, label=int(i))
        #for color, i in zip(colors, target_values):
        #    plt.scatter(X_r2[y_train == i, 0], X_r2[y_train == i, 1], X_r2[y_train == i, 2], alpha=.8, color=color, label=int(i))
        #plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('LDA of simile dataset')
        plt.show()


    #LDA - Linear Discriminant Analysis
    def lda_kfoldCrossValidation(self):
        s = Classification_simile  # this class
        x_train_list, x_test_list, y_train_list, y_test_list, target_values = s.readAndSplitKFoldsData('self', 10)
        lda = []
        x_d2_list = []
        #for a in range(1):
        index = -1
        precision = []
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            index += 1
            l = LinearDiscriminantAnalysis(solver='svd', n_components=2)
            #l = QuadraticDiscriminantAnalysis(store_covariances=True)
            lda.append(l)
            x_d2 = lda[index].fit(x_train, y_train).transform(x_train)
            x_d2_list.append(x_d2)
            y_pred = lda[index].predict(x_test)
            # print(y_pred)
            t = 0
            f = 0
            for y, y_t in zip(y_pred, y_test):
                if (y == y_t):
                    t += 1
                    #print(str(y) + " _ " + str(y_t))
                else:
                    f += 1
                    #print(str(y) + " _ " + str(y_t) + ' Error')
            print(str(t) + " " + str(f))
            print(str(t / (t + f)))
            print(classification_report(y_test, y_pred))
            precision.append(t / (t + f))
            #X_r2 = x_d2_list[0] #np.mean(x_d2_list, axis=0)
            colors = ['magenta', 'turquoise', 'brown',
                  'red', 'black', 'blue',
                  'pink', 'green', 'orange',
                  'yellow']
            for color, i in zip(colors, target_values):
                plt.scatter(x_d2[y_train == i, 0], x_d2[y_train == i, 1], alpha=.8, color=color, label=int(i))
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('LDA of simile dataset')
            plt.show()
        print(np.mean(precision))


    def readAndSplitKFoldsData(self, folds):
        feature_names, vector_space = vs3.VectorSpace_v3.numericalVectorSpace("self", main.filenames)
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
        return x_train_list, x_test_list, y_train_list, y_test_list, target_values



    def readAndSplitData(self, training_fraction):
        feature_names, vector_space = vs3.VectorSpace_v3.numericalVectorSpace("self", main.filenames)
        x = np.array(vector_space)
        #x = vector_space[0:, 0:]
        x = x.astype(float)
        target_values = []
        #for y_target
        for tn in x[0:, 0]:
            if not (tn in target_values):
                target_values.append(tn)
        #print(target_values)
        random.shuffle(x)
        l = len(x)
        training_len = int(l*training_fraction)
        x_train = x[:training_len, 1:]
        x_test = x[training_len:, 1:]
        y_train = x[:training_len, 0]  #target
        y_test = x[training_len:, 0]   #target
        return x_train, x_test, y_train, y_test, target_values, feature_names



Classification_simile()
