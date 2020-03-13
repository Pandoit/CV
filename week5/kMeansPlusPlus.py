"""
2D data set clustering
300 x 2 array referred to Machine learning course from Coursera
"""

import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.io import loadmat


def get_closest_dist(data, centro):
    """
    Get closest distance for centroid calculation
    :param data: dataset
    :param centroids:
    :return min_dist: minimum distance between centroid and data
    """
    min_dist = np.inf
    for i in range(centro.shape[0]):
        dist = np.linalg.norm((data - centro[i, :]))
        if dist < min_dist:
            min_dist = dist

    return min_dist


def dist_opt_initialize(data, K):
    """
    Initialize optimized centroids
    :param data:
    :param K:
    :return optimized_centroids:
    """
    m, n = data.shape
    data_i = np.random.choice(m, 1)
    optimized_centroids = data[data_i, :]
    d = np.zeros((1, m))
    for _ in range(1, K):
        total = 0.0
        for a in range(m):
            d[0, a] = get_closest_dist(data[a, :], optimized_centroids)
            total += d[0, a]
        total *= random.random()
        for b in range(m):
            total -= d[0, b]
            if total > 0:
                continue
            optimized_centroids = np.append(optimized_centroids, data[b, :].reshape(1, 2), axis=0)

            break

    return optimized_centroids


def find_closest_centroids(data, centroid):
    """
    computes the centroids for every data
    :param data: dataset
    :param centroids:
    :return idx: the closest centroids in idx for data
    """
    K = centroid.shape[0]
    idx = np.zeros((X.shape[0], 1))
    distance = np.zeros((K, 1))
    for i in range(data.shape[0]):
        for j in range(K):
            distance[j, 0] = np.sum((data[i, :]-centroid[j, :]) ** 2)
        val, ind = distance.min(0), distance.argmin(0)
        idx[i, 0] = ind

    return idx


def compute_centroids(data, index, K):
    """

    :param data:
    :param index:
    :param K:
    :return:
    """
    centroid = np.zeros((K, data.shape[1]))
    for i in range(K):
        temp = np.nonzero(index == i)
        centroid[i, :] = np.sum(data[temp[0], :], axis=0) / data[temp[0], :].shape[0]

    return centroid


def kmeans_plusplus(data, initial_centroid, max_iters):
    """
    kmeans++ main (same as classical kmeans)
    :param data: training set
    :param initial_centroid: computed centroids
    :param max_iters: total number of interactions of K-Means to execute
    :return centroid:
    :return index:
    """

    K = initial_centroid.shape[0]
    centroid = initial_centroid
    previous_centroid = centroid
    index = np.zeros((data.shape[0], 1))

    # Initialize plot
    plt.figure()
    plt.ion()

    for i in range(max_iters):
        index = find_closest_centroids(data, centroid)
        centroid = compute_centroids(data, index, K)

        # Plot result
        plt.clf()
        label_0 = np.nonzero(index == 0)[0]
        plt.scatter(data[label_0, 0], data[label_0, 1], color='blue', label='cluster1')
        label_1 = np.nonzero(index == 1)[0]
        plt.scatter(data[label_1, 0], data[label_1, 1], color='green', label='cluster2')
        label_2 = np.nonzero(index == 2)[0]
        plt.scatter(data[label_2, 0], data[label_2, 1], color='red', label='cluster3')
        plt.title('Final result by KMeans++', )
        plt.legend(loc='upper right')
        plt.pause(0.5)

    plt.ioff()
    plt.show()

    return centroid, index


if __name__ == "__main__":

    # Load data: 300 * 2 array
    example = loadmat('data.mat')
    print(example.keys())
    X = example['X']  # find right key
    print(X)

    # Plot dataset
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


    # define K clusters and max iterations
    K = 3
    maxIters = 10

    # Select an initial set of centroids
    initial_centroids = dist_opt_initialize(X, K)

    centroids, idx = kmeans_plusplus(X, initial_centroids, maxIters)




