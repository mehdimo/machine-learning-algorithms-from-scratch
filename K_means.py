"""
Algorithm
1) choose k random centroids
2) repeat until converged or reach the max_iter
    2-1) assign each data point to the closest centroid
    2-1) update the centroids to the mean of the clusters
    2-3) check for convergence, if no change in centroids, terminate
"""
from random import choice
import math

class KMeans:
    def __init__(self, k=3, max_iter=20, threshold=0.001):
        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold
        self.centroids = []
        self.clusters = []

    def fit(self, Xtrain):
        dim = len(Xtrain[0])
        self.centroids = [choice(Xtrain) for _ in range(self.k)]

        for iter in range(self.max_iter):
            # Assign points to centroids
            self.clusters = [[] for k in range(self.k)]
            for i, row in enumerate(Xtrain):
                dists = []
                for j in range(self.k):
                    dist = self.distance(row, self.centroids[j])
                    dists.append(dist)
                closes_index = dists.index(min(dists))
                self.clusters[closes_index].append(row)
            # update centroid
            new_centroids = []
            for cluster, points in enumerate(self.clusters):
                if points:
                    mean_point = []

                    for i in range(dim):
                        p = 0
                        for point in points:
                            p += point[i]
                        p = p / len(points)
                        mean_point.append(p)

                    new_centroids.append(mean_point)
                else:
                    new_centroids.append(self.centroids[cluster])

            shift = 0

            for m in range(self.k):
                diff = self.distance(new_centroids[m], self.centroids[m])
                shift = max(shift, diff)
            if shift <= self.threshold:
                print(f"Converged with max shift {shift}")
                return
            self.centroids = new_centroids
        print(f"finished {self.max_iter} iterations.")

    def classify(self, Xtest):
        return [self.classify_single(x) for x in Xtest]

    def classify_single(self, x):
        dists = []
        for centroid in self.centroids:
            d = self.distance(centroid, x)
            dists.append(d)
        closest = dists.index(min(dists))
        return closest


    def distance(self, a, b):
        if len(a) == len(b):
            sum_ = 0
            for i in range(len(a)):
                sum_ += ((a[i]-b[i]) ** 2)
            return math.sqrt(sum_)
        else:
            raise ValueError("Length mismatch")

if __name__ == "__main__":
    # Simple 2D data
    data = [
        [1, 2], [1.5, 1.8], [1.2, 2.1],     # cluster ~0
        [5, 8], [6, 7.5], [5.5, 8.2],       # cluster ~1
        [9, 1], [8.5, 1.2], [9.2, 0.8]    # cluster ~2
    ]

    kmeans = KMeans(k=3, max_iter=50, threshold=0.001)
    kmeans.fit(data)

    print("Final centroids:")
    for i, c in enumerate(kmeans.centroids):
        print(f"  Cluster {i}: { [round(x,2) for x in c] }")

    print("\nClusters:")
    for i, cluster in enumerate(kmeans.clusters):
        print(i, cluster)
