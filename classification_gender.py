from matplotlib import style
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
import main
import vectorSpace_gender as vsg
style.use("ggplot")
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import time
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import neighbors
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer



#classification and dimensionality reduction
class classification_gender:
    def __init__(self):
        start = time.time()
        #classification_gender.lda_evaluation('self', 2)
        classification_gender.lda_plot_2d_3d('self', 2)
        #classification_gender.classifiers_evaluation('self', 3)

        end = time.time()
        print("\n" + str(round((end - start), 3)) + " sec")



    # LDA - Linear Discriminant Analysis
    def lda_evaluation(self, numOfGender):
        cg = classification_gender
        x_train_list, x_test_list, y_train_list, y_test_list, target_values = \
            classification_gender.readAndSplitKFoldsData('self', 10, numOfGender)
        lda = []
        x_d2_list = []
        # for a in range(1):
        index = -1
        accuracy = []
        precision_list = []
        recall_list = []
        fscore_list = []
        suport_list = []
        labels_list = []
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            index += 1
            y_pred = cg.lda_classifier(self, x_train, y_train, x_test)
            #y_pred = cg.svm_classifier(self, x_train, y_train, x_test)
            #y_pred = cg.sgd_classifier(self, x_train, y_train, x_test)
            #y_pred = cg.kNeighbors_classifier(self, x_train, y_train, x_test, n_neighbors=30)
            #y_pred = cg.DT_classifier(self, x_train, y_train, x_test)
            #y_pred = cg.multiLayerPerceptron_classifier(self, x_train, y_train, x_test)
            #y_pred = cg.RF_classifier(self, x_train, y_train, x_test)
            t = 0
            f = 0
            for y, y_t in zip(y_pred, y_test):
                if (y == y_t):
                    t += 1
                    #print(str(y) + " _ " + str(y_t))
                else:
                    f += 1
                    #print(str(y) + " _ " + str(y_t) + ' Error')
            print(classification_report(y_test, y_pred))
            print("True: " + str(t) + " False: " + str(f))
            print("accuracy: " + str(round((t / (t + f)), 3)) + "\n")
            accuracy.append(t / (t + f))
            labels = unique_labels(y_test, y_pred)
            labels_list.append(labels)
            print(labels)
            pr, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)
            precision_list.append(pr)
            recall_list.append(rec)
            fscore_list.append(fs)
            suport_list.append(sup)
        labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support = \
            classification_gender.meanOfLists(self, labels_list, precision_list, recall_list, fscore_list, suport_list)
        labels = ["1_M", "2_F", "3_N"]
        if numOfGender == 2:
            labels = ["1_M/F", "2_N"]
        print('%-14s%-14s%-14s%-14s%-14s' % ("Gender", "Precision", "Recall", "F1-score", "Support"))
        #labels = labels_list[0]
        for l, p, r, f, s in zip(labels, precision, recall, fscore, support):
            tuple = (l, round(p, 3), round(r, 3), round(f, 3), int(round(s)))
            print('%-14s%-14s%-14s%-14s%-14s' % tuple)
        tuple = ('\nAvg/Total', round(avg_precision,3),round(avg_recall,3), round(avg_fscore,3), int(round(total_support)))
        print('%-14s%-14s%-14s%-14s%-14s' % tuple)
        print("avg accuracy: " + str(round(np.mean(accuracy),3)))





    def lda_plot_2d_3d(self, numOfGender):
        cg = classification_gender
        figure_number = 0
        for i in range(3):
            figure_number +=2
            x_train, x_test, y_train, y_test, target_values = classification_gender.readAndSplitData('self', 1)

            x_d2, x_d3 = cg.LSA('self', x_train)
            #x_d2, x_d3 = cg.LDA(self, x_train, y_train)


            colors = ['blue', 'orange', 'green']
            markers = ['o', 'P', 'X']
            labels = ["1_M", "2_F", "3_N"]
            if numOfGender == 2:
                colors = ['blue', 'orange']
                markers = ['P', 'X']
                labels = ["1_M/F", "2_N"]
            fig = plt.figure(figure_number-1)
            ax1 = fig.add_subplot(111)
            for color, i, m in zip(colors, target_values, markers):
                ax1.scatter(x_d2[y_train == i, 0], x_d2[y_train == i, 1], alpha=.8, color=color, marker=m, label=labels[int(i)-1])
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('Gender (2D)')
            #plt.show()
            fig2 = plt.figure(figure_number)
            ax2 = fig2.add_subplot(111, projection='3d')
            for color, i, m in zip(colors, target_values, markers):
                ax2.scatter(x_d3[y_train == i, 0], x_d3[y_train == i, 1], x_d3[y_train == i, 2], alpha=.8, c=color,
                           marker=m, label=labels[int(i)-1])
                plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('Gender (3D)')
            plt.show()



    # Latent Semantic Analysis
    def LSA(self, x_train):
            svd = TruncatedSVD(n_components=2)
            normalizer = Normalizer(copy=False)
            lsa_d2 = make_pipeline(svd, normalizer)
            x_d2 = lsa_d2.fit_transform(x_train)
            svd = TruncatedSVD(n_components=3)
            lsa_d3 = make_pipeline(svd, normalizer)
            x_d3 = lsa_d3.fit_transform(x_train)
            return x_d2, x_d3

    # Linear Discriminant Analysis
    def LDA(self, x_train, y_train):
            lda_d2 = LinearDiscriminantAnalysis(solver='svd', n_components=2)
            x_d2 = lda_d2.fit(x_train, y_train).transform(x_train)
            lda_d3 = LinearDiscriminantAnalysis(solver='svd', n_components=3)
            x_d3 = lda_d3.fit(x_train, y_train).transform(x_train)
            return x_d2, x_d3


    def lda_classifier(self, x_train, y_train, x_test):
       lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)
       # l = LinearDiscriminantAnalysis(solver='svd', n_components=2)
       # l = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.8, n_components=2)
       lda.fit(x_train, y_train)  # .transform(x_train)
       y_pred = lda.predict(x_test)
       return y_pred


    def svm_classifier(self, x_train, y_train, x_test):
       clf = svm.LinearSVC()
       clf.fit(x_train, y_train)  # .transform(x_train)
       y_pred = clf.predict(x_test)
       return y_pred

    def sgd_classifier(self, x_train, y_train, x_test):
       clf = SGDClassifier(loss="hinge", penalty="l2")
       clf.fit(x_train, y_train)  # .transform(x_train)
       y_pred = clf.predict(x_test)
       return y_pred


    def kNeighbors_classifier(self, x_train, y_train, x_test, n_neighbors=15):
        clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred

    def DT_classifier(self, x_train, y_train, x_test):
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred

    def multiLayerPerceptron_classifier(self, x_train, y_train, x_test):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 6), random_state=1)
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred

    def RF_classifier(self, x_train, y_train, x_test):
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features='auto', random_state=0)
        clf.fit(x_train, y_train)  # .transform(x_train)
        y_pred = clf.predict(x_test)
        return y_pred



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





    def readAndSplitKFoldsData(self, folds, numOfGenders):
        feature_names, vector_space = vsg.VectorSpace_gender.numericalVectorSpace_gender("self", main.filenames, numOfGender=numOfGenders)
        vector_space = np.array(vector_space)
        x = vector_space[1:, 0:]
        np.random.shuffle(x)
        x = x.astype(float)
        #print(x[0:, 0])
        target_values = []
        #for y_target
        for tn in x[1:, 1]: #gender index
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
                x_t.append(x[i, 2:])
                y_t.append(x[i, 1])
            x_train_list.append(x_t)
            y_train_list.append(y_t)  #target
            x_t = []
            y_t = []
            for i in test:
                x_t.append(x[i, 2:])
                y_t.append(x[i, 1])
            x_test_list.append(x_t)
            y_test_list.append(y_t)  #target
        return x_train_list, x_test_list, y_train_list, y_test_list, target_values


    def readAndSplitData(self, training_fraction):
        feature_names, vector_space = vsg.VectorSpace_gender.numericalVectorSpace_gender("self", main.filenames, 2)
        vector_space = np.array(vector_space)
        x = vector_space[1:, 0:]
        x = x.astype(float)
        target_values = []
        #for y_target
        for tn in x[1:, 0]:
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
        return x_train, x_test, y_train, y_test, target_values

    def classifiers_evaluation(self, numOfGender):
        cg = classification_gender
        x_train_list, x_test_list, y_train_list, y_test_list, target_values = \
            classification_gender.readAndSplitKFoldsData('self', 10, numOfGender)
        lda = []
        x_d2_list = []
        # for a in range(1):
        index, precision_list, recall_list, fscore_list, suport_list, labels_list = classification_gender.init('self')
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            index += 1
            y_pred = cg.lda_classifier(self, x_train, y_train, x_test)
            labels = unique_labels(y_test, y_pred)
            labels_list.append(labels)
            pr, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)
            precision_list.append(pr)
            recall_list.append(rec)
            fscore_list.append(fs)
            suport_list.append(sup)
            labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support = \
                classification_gender.meanOfLists(self, labels_list, precision_list, recall_list, fscore_list,
                                                  suport_list)
        print('%-14s%-14s%-14s%-14s%-14s' % ("Gender", "Precision", "Recall", "F1-score", "Support"))
        tuple = (
                '\nLDA', round(avg_precision, 3), round(avg_recall, 3), round(avg_fscore, 3),
                int(round(total_support)))
        print('%-14s%-14s%-14s%-14s%-14s' % tuple)
        index, precision_list, recall_list, fscore_list, suport_list, labels_list = classification_gender.init('self')
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            index += 1
            y_pred = cg.svm_classifier(self, x_train, y_train, x_test)
            labels = unique_labels(y_test, y_pred)
            labels_list.append(labels)
            pr, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)
            precision_list.append(pr)
            recall_list.append(rec)
            fscore_list.append(fs)
            suport_list.append(sup)
        labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support = \
                classification_gender.meanOfLists(self, labels_list, precision_list, recall_list, fscore_list,
                                                  suport_list)
        tuple = (
            'SVM', round(avg_precision, 3), round(avg_recall, 3), round(avg_fscore, 3), int(round(total_support)))
        print('%-14s%-14s%-14s%-14s%-14s' % tuple)

        index, precision_list, recall_list, fscore_list, suport_list, labels_list = classification_gender.init('self')
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            index += 1
            y_pred = cg.sgd_classifier(self, x_train, y_train, x_test)
            labels = unique_labels(y_test, y_pred)
            labels_list.append(labels)
            pr, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)
            precision_list.append(pr)
            recall_list.append(rec)
            fscore_list.append(fs)
            suport_list.append(sup)
        labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support = \
                classification_gender.meanOfLists(self, labels_list, precision_list, recall_list, fscore_list,
                                                  suport_list)
        tuple = (
                'SGD', round(avg_precision, 3), round(avg_recall, 3), round(avg_fscore, 3), int(round(total_support)))
        print('%-14s%-14s%-14s%-14s%-14s' % tuple)

        index, precision_list, recall_list, fscore_list, suport_list, labels_list = classification_gender.init('self')
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            index += 1
            y_pred = cg.kNeighbors_classifier(self, x_train, y_train, x_test, n_neighbors=30)
            labels = unique_labels(y_test, y_pred)
            labels_list.append(labels)
            pr, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)
            precision_list.append(pr)
            recall_list.append(rec)
            fscore_list.append(fs)
            suport_list.append(sup)
        labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support = \
                classification_gender.meanOfLists(self, labels_list, precision_list, recall_list, fscore_list,
                                                  suport_list)
        tuple = (
                'kNeighbors', round(avg_precision, 3), round(avg_recall, 3), round(avg_fscore, 3),
                int(round(total_support)))
        print('%-14s%-14s%-14s%-14s%-14s' % tuple)

        index, precision_list, recall_list, fscore_list, suport_list, labels_list = classification_gender.init('self')
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            index += 1
            y_pred = cg.DT_classifier(self, x_train, y_train, x_test)
            labels = unique_labels(y_test, y_pred)
            labels_list.append(labels)
            pr, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)
            precision_list.append(pr)
            recall_list.append(rec)
            fscore_list.append(fs)
            suport_list.append(sup)
        labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support = \
                classification_gender.meanOfLists(self, labels_list, precision_list, recall_list, fscore_list,
                                                  suport_list)
        tuple = (
                'DT', round(avg_precision, 3), round(avg_recall, 3), round(avg_fscore, 3),
                int(round(total_support)))
        print('%-14s%-14s%-14s%-14s%-14s' % tuple)

        index, precision_list, recall_list, fscore_list, suport_list, labels_list = classification_gender.init('self')
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            index += 1
            y_pred = cg.multiLayerPerceptron_classifier(self, x_train, y_train, x_test)
            labels = unique_labels(y_test, y_pred)
            labels_list.append(labels)
            pr, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)
            precision_list.append(pr)
            recall_list.append(rec)
            fscore_list.append(fs)
            suport_list.append(sup)
        labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support = \
                classification_gender.meanOfLists(self, labels_list, precision_list, recall_list, fscore_list,
                                                  suport_list)
        tuple = (
                'MLP', round(avg_precision, 3), round(avg_recall, 3), round(avg_fscore, 3),
                int(round(total_support)))
        print('%-14s%-14s%-14s%-14s%-14s' % tuple)


        index, precision_list, recall_list, fscore_list, suport_list, labels_list = classification_gender.init('self')
        for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
            index += 1
            y_pred = cg.RF_classifier(self, x_train, y_train, x_test)
            labels = unique_labels(y_test, y_pred)
            labels_list.append(labels)
            pr, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)
            precision_list.append(pr)
            recall_list.append(rec)
            fscore_list.append(fs)
            suport_list.append(sup)
        labels, precision, recall, fscore, support, avg_precision, avg_recall, avg_fscore, total_support = \
                classification_gender.meanOfLists(self, labels_list, precision_list, recall_list, fscore_list,
                                                  suport_list)
        tuple = (
                'RF', round(avg_precision, 3), round(avg_recall, 3), round(avg_fscore, 3),
                int(round(total_support)))
        print('%-14s%-14s%-14s%-14s%-14s' % tuple)




    def init(self):
        return -1, [], [], [], [], []


classification_gender()
