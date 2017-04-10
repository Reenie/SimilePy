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

class clustering:
    def __init__(self):
        # clustering.clustering_test("self")
        # clustering.test2("self")
        for a in range(10):
            clustering.lda('self')

        #iris = datasets.load_iris()
        #print(iris.data)

        #print(iris.target)
        #print(iris.target_names)

        #def som_test(self):


    #LDA - Linear Discriminant Analysis
    def lda(self):
        x_train, x_test, y_train, y_test, target_values = clustering.readAndSplitData('self', 0.90)
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
            print(str(y) + " _ " + str(y_t))
            if(y == y_t):
                t += 1
            else:
                f += 1
        print(str(t) + " " + str(f))
        print(str(t/(t+f)))
        plt.show()





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
