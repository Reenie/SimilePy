from matplotlib import style
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift  # as ms
import main
import vectorSpace
import vectorSpace_v2 as vs2
style.use("ggplot")
import numpy as np
import matplotlib
import random
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

class clustering:
    def __init__(self):
        # clustering.clustering_test("self")
        # clustering.test2("self")
        #for a in range(10):
        clustering.lda_kfoldCrossValidation('self')

        #iris = datasets.load_iris()
        #print(iris.data)

        #print(iris.target)
        #print(iris.target_names)

        #def som_test(self):

    #kf = KFold(n_splits=2)
    #for train, test in kf.split(X):


    #LDA - Linear Discriminant Analysis
    def lda(self):
        x_train, x_test, y_train, y_test, target_values = clustering.readAndSplitData('self', 1)
        lda = LinearDiscriminantAnalysis(n_components=2)
        x_d2_results = []
        #for a in range(1):
        X_r2 = lda.fit(x_train, y_train).transform(x_train)
        #x_d2_results.append(x_r2)
        #X_r2 = np.mean(x_d2_results, axis=0)

        colors = ['magenta', 'turquoise', 'brown',
                  'red', 'black', 'blue',
                  'pink', 'green', 'orange',
                  'yellow']
        for color, i in zip(colors, target_values):
            plt.scatter(X_r2[y_train == i, 0], X_r2[y_train == i, 1], alpha=.8, color=color, label=int(i))
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('LDA of simile dataset')

        y_pred = lda.predict(x_test)
        #print(y_pred)
        t = 0
        f = 0
        for y, y_t in zip(y_pred, y_test):

            if(y == y_t):
                t += 1
                print(str(y) + " _ " + str(y_t))
            else:
                f += 1
                print(str(y) + " _ " + str(y_t) + 'Error')
        print(str(t) + " " + str(f))
        print(str(t/(t+f)))
        plt.show()



    #LDA - Linear Discriminant Analysis
    def lda_kfoldCrossValidation(self):
        x_train_list, x_test_list, y_train_list, y_test_list, target_values = clustering.readAndSplitKFoldsData('self', 10)
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
        feature_names, vector_space = vs2.VectorSpace.numericalVectorSpace("self", main.filenames)
        vector_space = np.array(vector_space)
        x = vector_space[1:, 0:]
        np.random.shuffle(x)
        x = x.astype(float)
        #print(x[0:, 0])
        target_values = []
        #for y_target
        for tn in x[1:, 0]:
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
        feature_names, vector_space = vs2.VectorSpace.numericalVectorSpace("self", main.filenames)
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







    def test3(self):
        feature_names, vector_space = vs2.VectorSpace.numericalVectorSpace("self", main.filenames)
        vector_space = np.array(vector_space)
        X= vector_space[1:, 1:]
        x_target = vector_space[1:,0]
        print(clustering.find_centers(X, 3))






    def clustering_test(self):
        feature_names, vector_space = vectorSpace.VectorSpace.numericalVectorSpace("self", main.filenames)
        X = np.array(vector_space)
        ms = MeanShift()
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        print(cluster_centers)
        n_clusters_ = len(np.unique(labels))
        print("Number of estimated clusters:", n_clusters_)
        colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y', 'm']
        print(colors)
        print(labels)
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(X)):
            ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 10], cluster_centers[:, 10],
                   marker="x", color='k', s=150, linewidths=5, zorder=10)
        matplotlib.pyplot.show()


    def test2(self):
        # x = [1, 5, 1.5, 8, 1, 9]
        # y = [2, 8, 1.8, 8, 0.6, 11]

        # plt.scatter(x, y)
        # plt.show()
        feature_names, vector_space = vectorSpace.VectorSpace.numericalVectorSpace("self", main.filenames)
        # X = np.array([[1, 2],[5, 8],[1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
        X = np.array(vector_space)

        kmeans = KMeans(n_clusters=4)
        kmeans.fit(X)

        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        print(centroids)
        print(labels)

        # colors = ["g.", "r.", "c.", "y."]
        colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y', 'm']

        for i in range(len(X)):
            print("coordinate:", X[i], "label:", labels[i])
            matplotlib.pyplot.plot(X[i][0], X[i][1], colors[labels[i]], markersize=100)

        matplotlib.pyplot.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)

        matplotlib.pyplot.show()


        ### For the purposes of this example, we store feature data from our
        ### dataframe `df`, in the `f1` and `f2` arrays. We combine this into
        ### a feature matrix `X` before entering it into the algorithm.
        # f1 = df['Distance_Feature'].values
        # f2 = df['Speeding_Feature'].values
        # X = np.matrix(zip(f1, f2))
        # kmeans = KMeans(n_clusters=2).fit(X)


clustering()
