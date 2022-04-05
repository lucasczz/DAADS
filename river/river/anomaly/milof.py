# Milof - This is the Fast Memory Efficient Local Outlier Detection algorithm generated based on TKDE paper.
# For more information, please check
# https://ieeexplore.ieee.org/document/7530918/

# Inputs:
# kpar: number of nearest neighbours
# dimension: dimension of data points
# buck: bucket size (memory limit)
# filepath: path of input data file
# num_k: number of clusters used in kmeans, streaming kmeans and weighted kmeans
# width: N/A

# Authors: Dingwen Tao (ustc.dingwentao@gmail.com), Xi Zhang (xizhang1@cs.stonybrook.edu), Shinjae Yoo (sjyoo@bnl.gov)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class Point:
    def __init__(self):
        self.kdist = []
        self.knn = []
        self.lrd = []
        self.lof = []
        # self.rdist = {}


class Cluster:
    def __init__(self):
        self.center = []
        # self.LS = []


def lof(datastream, kpar):
    # not sure to use euclidean or minkowski
    Points = Point()
    clf = LocalOutlierFactor(
        n_neighbors=kpar, algorithm="kd_tree", leaf_size=30, metric="euclidean"
    )
    clf.fit(datastream)
    Points.lof = [-x for x in clf.negative_outlier_factor_.tolist()]
    Points.lrd = clf._lrd.tolist()
    dist, ind = clf.kneighbors()
    Points.kdist = dist.tolist()
    Points.knn = ind.tolist()
    return Points


def dist(a, b):
    dist = np.linalg.norm(a - b)
    return dist


def union(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    list3 = list1 + list(set2 - set1)
    return list3


def set_diff(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    list3 = list(set1 - set2)
    return list3


def IncrementalLOF_Fixed(Points, datastream, points_c, Clusters, kpar, buck, width):
    i = datastream.shape[0]
    # print("******************* Processing Data Point", i-1, "*******************")

    nbrs = NearestNeighbors(n_neighbors=kpar, algorithm="brute", metric="euclidean")
    nbrs.fit(datastream[0 : i - 1, :])
    dist, ind = nbrs.kneighbors(datastream[i - 1, :].reshape(1, -1))
    Points.kdist = Points.kdist + dist.tolist()
    Points.knn = Points.knn + ind.tolist()
    # print("Points.kdist = ", Points.kdist)
    # print("Points.knn = ", Points.knn)

    # check the distances of the point i with the cluster centers and pick the nearest cluster
    dist = []
    for j in range(0, len(Clusters.center)):
        distCI = dist(Clusters.center[j], datastream[i - 1, :]) - width
        # if distCI <=0:
        # 	distCI=points_c.kdist[j]
        dist = dist + [distCI]
    # print("dist = ", dist)

    if len(dist):
        minval, ind = min((dist[j], j) for j in range(0, len(dist)))
        for j in range(0, len(Points.kdist[i - 1])):
            if minval < Points.kdist[i - 1][j]:
                Points.kdist[i - 1][j] = minval
                Points.knn[i - 1][j] = buck + ind
                if j < len(Points.kdist[i - 1]) - 1:
                    Points.kdist[i - 1][j + 1 :] = [minval] * (
                        len(Points.kdist[i - 1]) - j - 1
                    )
                    Points.knn[i - 1][j + 1 :] = [buck + ind] * (
                        len(Points.kdist[i - 1]) - j - 1
                    )
                break

    rNN = []
    for k in range(0, i - 1):
        distance = dist(datastream[k, :], datastream[i - 1, :])
        # print ("distance between point", k, "and point", i-1, "=", distance)
        if Points.kdist[k][-1] >= distance:
            for kk in range(0, len(Points.knn[k])):
                if distance <= Points.kdist[k][kk]:
                    if kk == 0:
                        Points.knn[k] = [i - 1] + Points.knn[k][kk:]
                        Points.kdist[k] = [distance] + Points.kdist[k][kk:]
                    else:
                        Points.knn[k] = (
                            Points.knn[k][0:kk] + [i - 1] + Points.knn[k][kk:]
                        )
                        Points.kdist[k] = (
                            Points.kdist[k][0:kk] + [distance] + Points.kdist[k][kk:]
                        )
                    break

            for kk in range(kpar, len(Points.knn[k])):
                # print(Points.kdist[k][kk], Points.kdist[k][kpar-1])
                if Points.kdist[k][kk] != Points.kdist[k][kpar - 1]:
                    del Points.kdist[k][kk:]
                    del Points.knn[k][kk:]
                    break

            rNN = rNN + [k]
    # print("rNN = ", rNN)
    # print("Points.kdist = ", Points.kdist)
    # print("Points.knn = ", Points.knn)

    # update the updatelrd set
    updatelrd = rNN
    if len(rNN) > 0:
        for j in rNN:
            for ii in Points.knn[j]:
                if ii < len(Points.knn) and ii != i - 1 and j in Points.knn[ii]:
                    updatelrd = union(updatelrd, [ii])
    # print("updatelrd = ", updatelrd)

    # lrd-i
    rdist = 0
    for p in Points.knn[i - 1]:
        if p > buck - 1:
            rdist = rdist + max(
                dist(datastream[i - 1, :], Clusters.center[p - buck]) - width,
                points_c.kdist[p - buck],
            )
        else:
            rdist = rdist + max(
                dist(datastream[i - 1, :], datastream[p, :]), Points.kdist[p][-1]
            )
    Points.lrd = Points.lrd + [1 / (rdist / len(Points.knn[i - 1]))]
    # print("Points.lrd = ", Points.lrd)

    # lrd neighbours
    updatelof = updatelrd
    if len(updatelrd) > 0:
        for m in updatelrd:
            rdist = 0
            for p in Points.knn[m]:
                if p > buck - 1:
                    rdist = rdist + max(
                        dist(datastream[m, :], Clusters.center[p - buck])
                        - width,
                        points_c.kdist[p - buck],
                    )
                else:
                    rdist = rdist + max(
                        dist(datastream[m, :], datastream[p, :]),
                        Points.kdist[p][-1],
                    )
            Points.lrd[m] = 1 / (rdist / len(Points.knn[m]))
            for k in range(0, len(Points.knn)):
                if k != m and Points.knn[k].count(m) > 0:
                    updatelof = union(updatelof, [k])
    # print("Points.lrd =", Points.lrd)
    # print("updatelof =", updatelof)

    # lof neighbours
    if len(updatelof) > 0:
        for l in updatelof:
            lof = 0
            for ll in Points.knn[l]:
                if ll > buck - 1:
                    lof = lof + points_c.lrd[ll - buck] / Points.lrd[l]
                else:
                    lof = lof + Points.lrd[ll] / Points.lrd[l]
            if l == len(Points.lof):
                Points.lof = Points.lof + [lof / len(Points.knn[l])]
            else:
                Points.lof[l] = lof / len(Points.knn[l])
    # print("Points.lof =", Points.lof)

    # lof-i
    lof = 0
    for ll in Points.knn[i - 1]:
        if ll > buck - 1:
            lof = lof + points_c.lrd[ll - buck] / Points.lrd[i - 1]
        else:
            lof = lof + Points.lrd[ll] / Points.lrd[i - 1]
    if i - 1 == len(Points.lof):
        Points.lof = Points.lof + [lof / len(Points.knn[i - 1])]
    else:
        Points.lof[i - 1] = lof / len(Points.knn[i - 1])

    # print("Points.knn =", Points.knn)
    # print("Points.lof =", Points.lof)
    # print("Points.lrd =", Points.lrd)
    # print("rNN = ", rNN)
    # print("updatelrd = ", updatelrd)
    # print("updatelof =", updatelof)

    return Points


def MILOF(datastream, num_k, kpar, buck, width):
    # read parameters from configuration file
    scaler = MinMaxScaler()
    scaler.fit(datastream)
    datastream = scaler.transform(datastream)
    n_points = datastream.shape[0]
    print("number of data points =", n_points)

    hbuck = int(buck / 2)  # half of buck
    kdist = []
    scores = []
    clusterLog = []
    clusters = Cluster()
    points_c = Point()
    points = lof(datastream[0 : kpar + 1, :], kpar)
    scores = scores + points.lof

    for i in range(0, kpar + 1):
        kdist = kdist + [points.kdist[i][-1]]

    # using direct Incremental lof for the first bucket/2
    for i in range(kpar + 2, hbuck + 1):
        points = IncrementalLOF_Fixed(
            points, datastream[0:i, :], points_c, clusters, kpar, buck, width
        )
        scores = scores + [points.lof[i - 1]]
        kdist = kdist + [points.kdist[i - 1][-1]]



    exit = False
    step = 0
    while not exit:
        for i in range(hbuck + 1, buck + 1):
            if i > datastream.shape[0]:
                exit = True
                break
            points = IncrementalLOF_Fixed(
                points, datastream[0:i, :], points_c, clusters, kpar, buck, width
            )
            scores = scores + [points.lof[i - 1]]
            kdist = kdist + [points.kdist[i - 1][-1]]


        if not exit:
            step = step + 1

            print(
                "*******************Step",
                step,
                ": processing data points",
                step * hbuck,
                "to",
                (step + 1) * hbuck,
                "*******************",
            )

            index_normal = list(range(0, hbuck))
            kmeans = KMeans(
                n_clusters=num_k, init="k-means++", max_iter=100
            )  # Considering precompute_distances for faster but more memory
            kmeans.fit(
                datastream[0:hbuck, :]
            )  # need to check how to configure to match result of matlab code
            center = kmeans.cluster_centers_
            clusterindex = kmeans.labels_


            rem_clust_lbl = list(range(0, num_k))
            lof_scores = []
            for itr in range(0, hbuck):
                lof_scores = lof_scores + [points.kdist[itr][-1]]
            lof_scores = np.array(lof_scores)
            lof_threshold = np.mean(lof_scores) + 3 * np.std(
                lof_scores
            )  # Not sure if calcuating for each i is necessary


            for kk in range(0, num_k):
                clusterMembers = np.where(clusterindex == kk)
                clusterMembersList = np.asarray(clusterMembers).flatten().tolist()
                if np.sum(lof_scores[clusterMembers] > lof_threshold) > 0.5 * len(
                    clusterMembersList
                ):
                    index_normal = set_diff(index_normal, clusterMembersList)
                    rem_clust_lbl = set_diff(rem_clust_lbl, [kk])
            clusterindex = clusterindex[index_normal]
            center = center[rem_clust_lbl, :]


            # make summerization of clusters
            for j in range(0, len(rem_clust_lbl)):
                num = np.sum(clusterindex == rem_clust_lbl[j])
                points_c.knn = points_c.knn + [num]
                c_kdist = c_lrd = c_lof = 0
                for k in np.array(index_normal)[
                    np.where(clusterindex == rem_clust_lbl[j])
                ]:
                    c_kdist = c_kdist + points.kdist[k][-1]
                    c_lrd = c_lrd + points.lrd[k]
                    c_lof = c_lof + points.lof[k]

                points_c.kdist = points_c.kdist + [c_kdist / num]
                points_c.lrd = points_c.lrd + [c_lrd / num]
                points_c.lof = points_c.lof + [c_lof / num]


            datastream = np.delete(datastream, range(0, hbuck), axis=0)
            del points.kdist[0:hbuck]
            del points.knn[0:hbuck]
            del points.lrd[0:hbuck]
            del points.lof[0:hbuck]


            # Merge clusterings and points_c
            initialClusters = len(clusters.center)
            if initialClusters > 0:
                old_center = np.array(clusters.center)
                cluster_num = max(old_center.shape[0], center.shape[0])
                initial_center = []
                if cluster_num == center.shape[0]:
                    for x in center.tolist():
                        initial_center.append(np.array(x))
                else:
                    for x in old_center.tolist():
                        initial_center.append(np.array(x))


                wkmeans = KPlusPlus(
                    cluster_num,
                    X=np.concatenate((old_center, center), axis=0),
                    c=points_c.knn[0 : old_center.shape[0] + center.shape[0]],
                    max_runs=5,
                    verbose=False,
                    mu=initial_center,
                )
                wkmeans.find_centers(method="++")
                # strmkmeans = skm.StrmKMmeans(x=np.concatenate((old_center, center), axis=0), Centroids=initial_center, Assign=points_c.knn[0:old_center.shape[0]+center.shape[0]], )

                merge_center = np.array(wkmeans.mu)
                mergedindex = wkmeans.cluster_indices
                cluster_num = len(merge_center)
                clusterLog = clusterLog + [cluster_num]

                # update points_c by using extra parameter pc
                pc = Point()
                for j in range(0, cluster_num):
                    num = np.sum(mergedindex == j)
                    pc_knn = pc_kdist = pc_lrd = pc_lof = 0
                    for k in np.asarray(np.where(mergedindex == j)).flatten().tolist():
                        pc_knn = pc_knn + points_c.knn[k]
                        pc_kdist = pc_kdist + points_c.knn[k] * points_c.kdist[k]
                        pc_lrd = pc_lrd + points_c.knn[k] * points_c.lrd[k]
                        pc_lof = pc_lof + points_c.knn[k] * points_c.lof[k]

                    pc.knn = pc.knn + [pc_knn]
                    pc.kdist = pc.knn + [pc_kdist / pc_knn]
                    pc.lrd = pc.knn + [pc_lrd / pc_knn]
                    pc.lof = pc.knn + [pc_lof / pc_knn]
                points_c = pc

                # update Clusters
                clusters.center = merge_center.tolist()
            else:
                clusters.center = center.tolist()

            # print ("Clusters.center = ", Clusters.center)

            # update the Points (knn) as old datastream is deleted
            for j in range(0, len(points.knn)):
                for k in range(0, len(points.knn[j])):
                    if points.knn[j][k] >= hbuck:
                        if points.knn[j][k] < buck:
                            points.knn[j][k] = points.knn[j][k] - hbuck
                        else:
                            points.knn[j][k] = (
                                mergedindex[points.knn[j][k] - buck] + buck
                            )
                    else:
                        indarr = np.where(np.array(index_normal) == points.knn[j][k])
                        ind = np.asarray(indarr).flatten().tolist()
                        if len(ind):
                            cLabel = clusterindex[indarr]
                            ci = (
                                np.asarray(np.where(np.array(rem_clust_lbl) == cLabel))
                                .flatten()
                                .tolist()[0]
                            )
                            if not initialClusters:
                                points.knn[j][k] = ci + buck
                            else:
                                points.knn[j][k] = (
                                    mergedindex[ci + initialClusters] + buck
                                )
                        else:
                            mindist = []
                            for kk in range(0, len(clusters.center)):
                                mindist = mindist + [
                                    dist(datastream[j, :], clusters.center[kk])
                                ]
                            mindistVal, ci = min(
                                (mindist[x], x) for x in range(0, len(mindist))
                            )
                            points.knn[j][k] = ci + buck
    return scores


# Weighted k-means algorithm
# Code: Olivia Guest (weighted k-means) and The Data Science Lab (k-means and k-means++)
# Algorithm: Bradley C. Love (weighted k-means)
# Original code for vanilla k-means and k-means++ can be found at:
# https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
# https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/

# Required argument:
# K       -- the number of clusters.
# Keyword arguments:
# X        -- the data (default None; thus auto-generated, see below);
# N        -- number of unique data points to generate (default: 0);
# c        -- number of non-unique points represented by a data point
#                 (default: None; to mean every data point is unique);
# alpha    -- the exponent used to calculate the scaling factor (default: 0);
# beta     -- the stickiness parameter used during time-averaging
#                 (default: 0);
# dist     -- custom distance metric for calculating distances between points
#                 (default: great circle distance);
# max_runs -- When to stop clustering, to avoid infinite loop (default: 200);
# label    -- offer extra information at runtime about what is being
#                 clustered (default: 'My Clustering');
# verbose  -- how much information to print (default: True).
# mu       -- seed clusters, i.e., define a starting state (default: None).
# max_diff -- maximum perceptible change between present and previous
#             centroids when checking if solution is stable (default: 0.001).

from __future__ import division, print_function

import random
import sklearn.datasets

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def euclidean(a, b):
    """An example of what could be used as a distance metric."""
    return np.linalg.norm(np.asarray(a) - np.asarray(b))


class WKMeans:
    # Class for running weighted k-means.
    def counted(f):
        """Decorator for returning number of times a function has been called.
        Code: FogleBird
        Source: http://stackoverflow.com/a/21717396;
        """

        def wrapped(*args, **kwargs):
            wrapped.calls += 1
            return f(*args, **kwargs)

        wrapped.calls = 0
        return wrapped

    def __init__(
        self,
        K,
        X=None,
        N=0,
        c=None,
        alpha=0,
        beta=0,
        dist=euclidean,
        max_runs=200,
        label="My Clustering",
        verbose=True,
        mu=None,
        max_diff=0.001,
    ):
        """Initialisation."""
        self.K = K
        if X is None:
            if N == 0:
                raise Exception(
                    "If no data is provided, \
                                 a parameter N (number of points) is needed"
                )
            else:
                self.N = N
                self.X = self._init_gauss(N)
        else:
            self.X = X
            self.N = len(X)
        # The coordinates of the centroids:
        self.mu = mu
        # We need to keep track of previous centroids.
        self.old_mu = None
        # What kind of initialisation we want, vanilla, seed, or k++ available:
        if self.mu is None:
            self.method = "random"
        else:
            self.method = "manual"
        # Numpy array of clusters containing their index and member items.
        self.clusters = None
        self.cluster_indices = np.asarray([None for i in self.X])
        # For scaling distances as a function of cluster size:
        # the power the cardinalities will be raised to;
        self.alpha = alpha
        # and to scale the distances between points and centroids, which is
        # initialised to 1/k for all clusters.
        self.scaling_factor = np.ones((self.K)) / self.K
        # The stickiness used within the time-averaging.
        self.beta = beta
        # How many counts are represented by a single data point:
        if c is None:
            self.counts_per_data_point = [1 for x in self.X]
        else:
            self.counts_per_data_point = c
        # How many counts are in each cluster:
        self.counts_per_cluster = [0 for x in range(self.K)]
        # Use max_runs to stop running forever in cases of non-convergence:
        self.max_runs = max_runs
        # How many runs so far:
        self.runs = 0
        # The distance metric to use, a function that takes a and b and returns
        # the distrance between the two:
        self.dist = dist
        # The maximum difference between centroids from one run to the next
        # which still counts as no change:
        self.max_diff = max_diff
        # A label, to print out while running k-means, e.g., to distinguish a
        # specific instance of k-means, etc:
        self.label = label
        # How much output to print:
        self.verbose = verbose

    def _init_gauss(self, N):
        """Create test data in which there are three bivariate Gaussians.
        Their centers are arranged in an equilateral triange, with the top
        Gaussian having double the density of the other two. This is tricky
        test data because the points in the top part is double that of the
        lower parts of the space, meaning that a typical k-means run will
        create unequal clusters, while a weighted k-means will attempt to
        balance data points betwen the clusters.
        """
        # Set up centers of bivariate Gaussians in a equilateral triangle.
        centers = [[0, 0], [1, 0], [0.5, np.sqrt(0.75)]]

        # The SDs:
        cluster_std = [0.3, 0.3, 0.3]

        # The number of points, recall we need double at the top point hence
        # 3/4 of points are being generated now.
        n_samples = int(np.ceil(0.75 * N))

        data, labels_true = sklearn.datasets.make_blobs(
            n_samples=n_samples, centers=centers, cluster_std=cluster_std
        )

        # Now to generate the extra data points for the top of the triangle:
        centers = [[0.5, np.sqrt(0.75)]]
        cluster_std = [0.3]
        # n_clusters = len(centers)
        extra, labels_true = sklearn.datasets.make_blobs(
            n_samples=int(0.25 * N), centers=centers, cluster_std=cluster_std
        )

        # Merge the points together to create the full dataset.
        data = np.concatenate((data, extra), axis=0)
        return data

    @counted
    def _cluster_points(self):
        """Cluster the points."""
        # Initialise the values for the clusters and their counts
        clusters = [[] for i in range(self.K)]
        counts_per_cluster = [0 for i in range(self.K)]

        #######################################################################
        # Firstly perform classical k-means, weighting the distances.
        #######################################################################
        for index, x in enumerate(self.X):
            # For each data point x, find the minimum weighted distance to
            # cluster i from point x.
            bestmukey = min(
                [
                    (i[0], self.scaling_factor[i[0]] * self.dist(x, self.mu[i[0]]))
                    for i in enumerate(self.mu)
                ],
                key=lambda t: t[1],
            )[0]
            # Add the data point x to the cluster it is closest to.
            clusters[bestmukey].append(x)
            counts_per_cluster[bestmukey] += self.counts_per_data_point[index]
            self.cluster_indices[index] = bestmukey

        # Update the clusters.
        clusters = [c for c in clusters if len(c)]
        scaling_factor = np.asarray(
            [self.scaling_factor[i] for i in range(self.K) if counts_per_cluster[i]]
        )
        counts_per_cluster = [num for num in counts_per_cluster if num]
        self.clusters = clusters
        self.scaling_factor = scaling_factor
        self.counts_per_cluster = counts_per_cluster
        self.K = len(self.clusters)

        #######################################################################
        # Secondly, calculate the scaling factor for each cluster.
        #######################################################################
        # Now that we have clusters to work with (at initialisation we don't),
        # we can calculate the scaling_factor per cluster. This calculates the
        # cardinality of the cluster raised to the power alpha, so it is purely
        # a function of the number of items in each cluster and the value of
        # alpha.
        scaling_factor = np.asarray(
            [
                self.counts_per_cluster[index] ** self.alpha
                for index, cluster in enumerate(self.clusters)
            ]
        )

        # Now we have all the numerators, divide them by their sum. This is
        # also known as the Luce choice share.
        scaling_factor = scaling_factor / np.sum(scaling_factor)

        # The scaling factors should sum to one here.
        # print 'Sum of luce choice shares:', np.around(np.sum(scaling_factor))
        # assert np.around(np.sum(scaling_factor)) == 1

        # Now we want to employ time-averaging on the scaling factor.
        scaling_factor = (1 - self.beta) * scaling_factor + (
            self.beta
        ) * self.scaling_factor

        # Update the scaling factors for the next time step.
        self.scaling_factor = scaling_factor

        # The scaling factors should sum to one here too.
        # print 'Sum of scaling factors:',\
        #         np.around(np.sum(self.scaling_factor))
        # assert np.around(np.sum(self.scaling_factor)) == 1

    def _reevaluate_centers(self):
        """Update the controids (aka mu) per cluster."""
        new_mu = []
        for k in self.clusters:
            # For each key, add a new centroid (mu) by calculating the cluster
            # mean.
            new_mu.append(np.mean(k, axis=0))

        # Update the list of centroids that we just calculated.
        self.mu = new_mu

    def _has_converged(self):
        """Check if the items in clusters have stabilised between two runs.
        This checks to see if the distance between the centroids is lower than
        a fixed constant.
        """
        diff = 1000
        if self.clusters:
            for clu in self.clusters:
                # For each clusters, check the length. If zero, we have a
                # problem, we have lost clusters.
                if len(clu) is 0:
                    raise ValueError(
                        "One or more clusters disappeared because"
                        " all points rushed away to other"
                        " cluster(s). Try increasing the"
                        " stickiness parameter (beta)."
                    )
            # Calculate the mean distance between previous and current
            # centroids.
            diff = 0
            for i in range(self.K):
                diff += self.dist(self.mu[i].tolist(), self.old_mu[i].tolist())
            diff /= self.K

        # Return true if the items in each cluster have not changed much since
        # the last time this was run:
        return diff < self.max_diff

    def find_centers(self, method="random"):
        """Find the centroids per cluster until equilibrium."""
        self.method = method
        X = self.X
        K = self.K
        # Previous centroids set to random values.
        self.old_mu = random.sample(list(X), K)

        if method == "random":
            # If method of initialisation is not k++, use random centeroids.
            self.mu = random.sample(X, K)

        while not self._has_converged() and self.runs < self.max_runs:
            # While the algorithm has neither converged nor been run too many
            # times:
            # a) keep track of old centroids;
            self.old_mu = self.mu
            # b) assign all points in X to clusters;
            self._cluster_points()
            # c) recalculate the centers per cluster.
            self._reevaluate_centers()
            self.runs += 1


class KPlusPlus(WKMeans):
    """Augment the WKMeans class with k-means++ capabilities."""

    def _dist_from_centers(self):
        """Calculate the distance of each point to the closest centroids."""
        cent = self.mu
        X = self.X
        D2 = np.array([min([self.dist(x, c) ** 2 for c in cent]) for x in X])
        self.D2 = D2

    def _choose_next_center(self):
        """Select the next center probabilistically."""
        self.probs = self.D2 / self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = random.random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return self.X[ind]

    def init_centers(self):
        """Initialise the centers."""
        self.mu = random.sample(list(self.X), 1)
        while len(self.mu) < self.K:
            self._dist_from_centers()
            self.mu.append(self._choose_next_center())
