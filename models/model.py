"""
Custom K-means clustering implementation with k-means++ initialization.
Provides Lloyd's algorithm with enhanced initialization and unlearning capabilities.
"""
import numpy as np


class MyKmeans(object):
    """K-means clustering with k-means++ initialization and Lloyd's iterations"""

    def __init__(self, k, termination='loss', max_iters=10, tol=1e-3):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.termination = termination
        self._init_placeholders()

    def _init_placeholders(self):
        self.loss = np.Infinity
        self.empty_clusters = []
        self.kpp_inits = []  # record the idx for the initial centroids
        self.centroids = None
        self.assignments = None
        self.quant_centroids = None
        self.cluster_sizes = None

    def run_kpp_only(self, X):
        self._set_data(X)
        self._init_centroids()
        self._assign_clusters()
        return self.centroids, self.assignments, self.loss

    def run(self, X, pre_init=False):
        """Full k-means clustering algorithm with Lloyd's iterations"""
        self._set_data(X)
        self._lloyd_iterations(pre_init)
        return self.centroids, self.assignments, self.loss

    def unlearn_check(self, del_idx):
        """Check if deletion requires retraining or just data removal"""
        assert len(self.kpp_inits) == self.k, "number of centroids is less than K"
        if del_idx in self.kpp_inits:
            return True
        
        # Update indices and remove data point
        for k_val in range(self.k):
            if self.kpp_inits[k_val] > del_idx:
                self.kpp_inits[k_val] -= 1
        self.n -= 1
        
        # Update loss and remove data
        tmp_norm = np.linalg.norm(self.data[del_idx, :] - 
                                  self.centroids[self.assignments[del_idx], :])
        self.loss -= tmp_norm**2
        self.data = np.delete(self.data, del_idx, 0)
        self.assignments = np.delete(self.assignments, del_idx)
        return False

    def _set_data(self, X):
        self.data = X
        self.n, self.d = X.shape

    def set_centroids(self, centroids):
        self.centroids = centroids

    def _lloyd_iterations(self, pre_init=False):
        if not pre_init:
            self._init_centroids()
        for _ in range(self.max_iters):
            loss_prev = self.loss
            centroids_prev = self.centroids
            self._assign_clusters(
            )  # get new loss and assignment after we re-assign the clusters
            self._assign_centroids()  # get new centroids
            prev = loss_prev if self.termination == 'loss' else centroids_prev
            if self._check_termination(prev):
                break

    def _check_termination(self, prev):
        if self.termination == 'loss':
            return (1 - self.loss / prev) < self.tol
        elif self.termination == 'center':
            return np.linalg.norm(self.centroids - prev) < self.tol
        else:
            return False

    def _init_centroids(self):
        """K-means++ initialization for better centroid selection"""
        first_idx = np.random.choice(self.n)
        self.kpp_inits.append(first_idx)
        self.centroids = self.data[first_idx, :]
        for _ in range(1, self.k):
            P = self._get_selection_prob()
            nxt_idx = np.random.choice(self.n, p=P)
            assert nxt_idx not in self.kpp_inits, "error during K-means++ initialization"
            self.kpp_inits.append(nxt_idx)
            self.centroids = np.vstack([self.centroids, self.data[nxt_idx, :]])

    def _get_selection_prob(self):
        """Compute selection probabilities based on squared distances to nearest centroid"""
        if len(self.centroids.shape) == 1:
            self.centroids = np.expand_dims(self.centroids, axis=0)
        D = np.zeros([self.n])
        for i in range(self.n):
            d = np.linalg.norm(self.data[i, :] - self.centroids, axis=1)
            D[i] = np.min(d)
        P = [dist**2 for dist in D]
        P = P / sum(P)
        return P

    def _assign_centroids(self):
        """Update centroids based on current cluster assignments"""
        self.centroids = np.zeros([self.k, self.d])
        c = np.zeros([self.k])  # Cluster sizes for normalization
        for i in range(self.n):
            a = self.assignments[i]
            c[a] += 1
            self.centroids[a, :] += self.data[i, :]

        for j in range(self.k):
            if j not in self.empty_clusters:
                self.centroids[j, :] = self.centroids[j, :] / c[j]

        for j in self.empty_clusters:
            self._reinit_cluster(j)
        self.empty_clusters = []

    def _assign_clusters(self):
        """Assign data points to nearest centroids"""
        assert (self.k, self.d) == self.centroids.shape, "Centers wrong shape"
        self.assignments = np.zeros([self.n]).astype(int)
        self.loss = 0
        for i in range(self.n):
            d = np.linalg.norm(self.data[i, :] - self.centroids, axis=1)
            self.assignments[i] = int(np.argmin(d))
            self.loss += np.min(d)**2
        self.empty_clusters = self._check_4_empty_clusters()

    def _check_4_empty_clusters(self):
        """Check for empty clusters in assignments"""
        empty_clusters = []
        for kappa in range(self.k):
            if len(np.where(self.assignments == kappa)[0]) == 0:
                empty_clusters.append(kappa)
        return empty_clusters

    def _reinit_cluster(self, j):
        """Reinitialize empty cluster using k-means++ selection"""
        P = self._get_selection_prob()
        j_prime = np.random.choice(self.n, p=P)
        if j_prime not in self.kpp_inits:
            self.kpp_inits.append(j_prime)
        self.centroids[j, :] = self.data[j_prime, :]
        return j_prime

    def quantize_centroids(self, quant_eps):
        self.quant_centroids = self.centroids / quant_eps
        self.quant_centroids = np.round(self.quant_centroids)
        self.quant_centroids = self.quant_centroids * quant_eps
        return self.quant_centroids

    def calculate_cluster_sizes(self):
        self.cluster_sizes = np.zeros([self.k], dtype=int)
        for i in range(self.n):
            a = self.assignments[i]
            self.cluster_sizes[a] += 1
        return self.cluster_sizes
