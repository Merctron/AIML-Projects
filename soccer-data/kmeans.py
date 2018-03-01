# A generalized implementation of the k-means clustering algorithm 
# using pandas

# Imports
from __future__ import division

import sys
import random
import math

DIFF_FUNC = {
    'EUCLIDEAN': 0,
    'MANHATTAN': 1
}

class KmeansConfig:
    def __init__(self, diff_cols, diff_labels=None, diff_func=0):
        self.diff_cols   = diff_cols
        self.diff_labels = diff_labels
        self.diff_func   = diff_func

class Kmeans:
    def __init__(self, config, k, data):
        """Initializes the kmeans class. data should be a pandas dataframe"""
        self.config      = config
        self.k           = k
        self.data        = data
        self.seeds       = []
        self.data_points = []

    def generateSeeds(self):
        """Generates seeds to use as initial centroid in the data"""
        seeds     = []
        rand_seed = 0

        while len(seeds) < self.k:
            rand_seed = random.randint(0, len(self.data.index))
            if rand_seed not in seeds:
                seeds.append(rand_seed)

        self.seeds = seeds

    def generateDataPoints(self):
        """Generate n-dimensional (n=len(config.diff_cols)) data points from the passed in data frame"""
        data_points = []
        for i in range(len(self.data.index)):
            data_points.append([])
            for j in range(len(self.config.diff_cols)):
                data_points[i].append(self.data.iloc[i][self.config.diff_cols[j]])

        self.data_points = data_points

    def eucDist(self, pt1, pt2):
        """Calculates euclidean distance between two data points"""
        sum = 0
        for i in range(len(pt1)):
            sum = sum + math.pow((pt1[i] - pt2[i]), 2)
        return math.sqrt(sum)

    def manDist(self, pt1, pt2):
        """Calculates manhattan distance between two data points"""
        sum = 0
        for i in range(len(pt1)):
            sum = sum + abs(pt1[i] - pt2[i])
        return sum

    def computeCentroids(self, centroids, old_clusters=None):
        """Runs an iteration to calculate assignment to a cluster"""
        
        # Step 1: Loop through all data points and assign to clusters
        clusters = []
        change_flag = False
        
        for i in range(self.k):
            clusters.append([])

        for i in range(len(self.data_points)):
            pastDist = sys.maxint
            clusterIndex = 0

            for j in range(len(centroids)):
                dist = sys.maxint
                if self.config.diff_func == DIFF_FUNC.EUCLIDEAN:
                    dist = self.eucDist(centroids[j], 
                        self.data_points[i])
                elif self.config.diff_func == DIFF_FUNC.MANHATTAN:
                    dist = self.manDist(centroids[j], 
                        self.data_points[i])

                if dist < pastDist:
                    pastDist = dist
                    clusterIndex = j

            clusters[clusterIndex].append(i)
            if old_clusters is not None:
                if i not in old_clusters[clusterIndex]:
                    change_flag = True
            else:
                change_flag = True

        # Step 2: Compute new centroids based on assignment
        new_centroids = []

        for i in range(len(clusters)):
            new_centroids.append([])

            for k in range(len(self.data_points[0])):
                new_centroids[i].append(0)

            for j in range(len(clusters[i])):
                for k in range(len(self.data_points[0])):
                    new_centroids[i][k] = new_centroids[i][k] + self.data_points[clusters[i][j]][k]

            for k in range(len(new_centroids[i])):
                new_centroids[i][k] = new_centroids[i][k] / len(clusters[i])

        return change_flag, new_centroids, clusters

    def cluster(self):
        """Runs clustering/centroid computation epochs until there is no assignment change"""

        # Step 0: Prepare data
        self.generateDataPoints()

        # Step 1: Generate random seeds to begin clustering with
        self.generateSeeds()
        centroids = []
        for i in range(len(self.seeds)):
            centroids.append(self.data_points[self.seeds[i]])

        # Step 2: Run computation epochs
        change_flag = True
        clusters    = None
        num_epochs  = 0
        while change_flag:
            change_flag, centroids, clusters = self.computeCentroids(centroids, clusters)
            num_epochs = num_epochs + 1
            print("Epoch: {0}\n".format(num_epochs))
            for i in range(len(centroids)):
                if self.config.diff_labels:
                    print("Centroid Labels: {0}".format(self.config.diff_labels))
                print("Centroid: {0}, Occupancy: {1}, Values: {2}". format(i, len(clusters[i]), centroids[i]))

        #print("Final Centroids: {0}\n".format(centroids))