

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
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist


class clustering:
    def __init__(self):
        # clustering.clustering_test("self")
        # clustering.test2("self")
        clustering.test3('self')

    # def som_test(self):


    def cluster_points(X, mu):
        clusters = {}
        for x in X:
            bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) \
                             for i in enumerate(mu)], key=lambda t: t[1])[0]
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        return clusters

    def reevaluate_centers(mu, clusters):
        newmu = []
        keys = sorted(clusters.keys())
        for k in keys:
            newmu.append(np.mean(clusters[k], axis=0))
        return newmu

    def has_converged(mu, oldmu):
        return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

    def find_centers(X, K):
        # Initialize to K random centers
        oldmu = random.sample(X, K)
        mu = random.sample(X, K)
        while not clustering.has_converged(mu, oldmu):
            oldmu = mu
            # Assign all points in X to clusters
            clusters = clustering.cluster_points(X, mu)
            # Reevaluate centers
            mu = clustering.reevaluate_centers(oldmu, clusters)
        return (mu, clusters)


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
