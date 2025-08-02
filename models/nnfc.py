# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2025-06-02 17:55:49
# @Last Modified by:   Yunbo
# @Last Modified time: 2025-07-04 13:04:33

# Optimized version with multi-processing and GPU acceleration support
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import pickle
import math
import os
from sklearn.cluster import DBSCAN
import random
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from typing import List, Tuple
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import multiprocessing as mp
from functools import partial
import time
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Try to use GPU acceleration if available
try:
    import cupy as cp
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    GPU_AVAILABLE = True
    # print("GPU acceleration available!")
except ImportError:
    GPU_AVAILABLE = False
    # print("GPU acceleration not available, using CPU")

def load_dataset(filepath):
    """Load dataset from pickle file"""
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset

def dist(a, b, ax=1):
    """Compute distance between two points"""
    return np.linalg.norm(a - b, axis=ax)

def SNN_optimized(k: int, nc: int, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized SNN clustering algorithm"""
    unassigned = -1
    n, d = data.shape

    # Use faster distance computation
    if GPU_AVAILABLE and n > 1000:
        try:
            data_gpu = cp.asarray(data)
            distance = cp.asnumpy(cp.linalg.norm(data_gpu[:, None] - data_gpu[None, :], axis=2))
        except:
            distance = squareform(pdist(data))
    else:
        distance = squareform(pdist(data))

    # Vectorized neighbor computation
    indexDistanceAsc = np.argsort(distance, axis=1)
    indexNeighbor = indexDistanceAsc[:, :k]

    # Optimized shared neighbor computation using broadcasting
    numSharedNeighbor = np.zeros([n, n], dtype=int)
    
    for i in range(n):
        for j in range(i):
            shared = np.intersect1d(indexNeighbor[i], indexNeighbor[j])
            numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = len(shared)

    # Vectorized similarity computation
    similarity = np.zeros([n, n])
    mask = numSharedNeighbor > 0
    
    for i in range(n):
        for j in range(i):
            if mask[i, j]:
                shared = np.intersect1d(indexNeighbor[i], indexNeighbor[j])
                if i in shared and j in shared:
                    distanceSum = np.sum(distance[i, shared] + distance[j, shared])
                    if distanceSum > 0:
                        similarity[i, j] = similarity[j, i] = numSharedNeighbor[i, j] ** 2 / distanceSum

    # Compute rho, delta, and gamma
    rho = np.sum(np.sort(similarity, axis=1)[:, -k:], axis=1)
    
    distanceNeighborSum = np.sum(distance[np.arange(n)[:, None], indexNeighbor], axis=1)
    indexRhoDesc = np.argsort(rho)[::-1]
    delta = np.full(n, np.inf)
    
    for i, a in enumerate(indexRhoDesc[1:], 1):
        for b in indexRhoDesc[:i]:
            delta[a] = min(delta[a], distance[a, b] * (distanceNeighborSum[a] + distanceNeighborSum[b]))
    
    delta[indexRhoDesc[0]] = np.max(delta[delta != np.inf]) if np.any(delta != np.inf) else 1.0
    
    gamma = rho * delta
    
    # Compute centroids and assignments
    indexAssignment = np.full(n, unassigned)
    indexCentroid = np.sort(np.argsort(gamma)[-nc:])
    indexAssignment[indexCentroid] = np.arange(nc)

    # Assignment process (simplified)
    queue = indexCentroid.tolist()
    while queue:
        a = queue.pop(0)
        for b in indexNeighbor[a]:
            if indexAssignment[b] == unassigned and numSharedNeighbor[a, b] >= k // 2:
                indexAssignment[b] = indexAssignment[a]
                queue.append(b)

    # Handle remaining unassigned points
    indexUnassigned = np.where(indexAssignment == unassigned)[0]
    while len(indexUnassigned) > 0:
        for idx in indexUnassigned:
            neighbors = indexDistanceAsc[idx, :k]
            assigned_neighbors = neighbors[indexAssignment[neighbors] != unassigned]
            if len(assigned_neighbors) > 0:
                # Assign to most common cluster among neighbors
                clusters, counts = np.unique(indexAssignment[assigned_neighbors], return_counts=True)
                indexAssignment[idx] = clusters[np.argmax(counts)]
        
        new_unassigned = np.where(indexAssignment == unassigned)[0]
        if len(new_unassigned) == len(indexUnassigned):
            # No progress made, assign remaining to random clusters
            indexAssignment[new_unassigned] = np.random.randint(0, nc, len(new_unassigned))
            break
        indexUnassigned = new_unassigned

    return indexCentroid, indexAssignment

def add_laplace_noise_vectorized(data, epsilon, sensitivity):
    """Vectorized Laplace noise addition"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise

# def nnfc_optimized(data_path, use_gpu=False,k1=None,k2=None):
#     """Optimized NNFC function"""

#     datapkl = load_dataset(data_path)
#     # eachlable = datapkl['eachlable']
#     # order = datapkl['order']
#     # print('clients',len(order))
#     data = np.array(datapkl['full_data'])
    
#     corepoints = []
    
#     for i_client in range(10):
#         lodata = datapkl["client_" + str(i_client)]

#         epsilon = 0.1
#         scale = 1 / epsilon
#         laplace_noise = np.random.laplace(loc=0.0, scale=scale, size=lodata.shape)


#         lodata_noisy = lodata + laplace_noise
    
    
#         # n_clusters = min(len(lodata_noisy) // 3, 50)
#         n_clusters = k1
        
        
#         cluster = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         centers = cluster.fit(lodata_noisy).cluster_centers_
    
#         corepoints.append(centers)
    
#     serverdata = np.concatenate(corepoints, axis=0)
#     label = datapkl['true_label']
#     cnum = len(set(label))
#     # print('cnum',cnum)
#     # k = min(5, len(serverdata) // 10)  # Adaptive k

#     k = k2
    
#     centroid, assignment = SNN_optimized(k, cnum, serverdata)

#     print(centroid.shape)
#     print(serverdata.shape)
        
#     finalcenter = []
#     ii = 0
#     for i in centroid:
#         finalcenter.append(serverdata[i])
#         ii = ii + 1

#     idx = []
#     for i in data:
#         simi = []
#         for j in finalcenter:
#             simi.append(np.linalg.norm(i - j))
#         idx.append(simi.index(min(simi)) + 1)

#     arr = np.array(idx)
#     ari = round(adjusted_rand_score(label, arr), 4)
#     nmi = round(normalized_mutual_info_score(label, arr), 4)
#     return ari,nmi,[n_clusters, cnum, k]
    
def nnfc_optimized(data_path, use_gpu=False, k1=None, k2=None,epsilon=None,seed=None):
    """Optimized NNFC function with centroid label checking"""
    datapkl = load_dataset(data_path)
    data = np.array(datapkl['full_data'])
    true_labels = np.array(datapkl['true_label'])  # Original labels for full data
    
    corepoints = []
    client_data_indices = []  # To track which data points belong to each client
    start_idx = 0

    np.random.seed(seed)

    
    # Process each client's data
    for i_client in range(10):
        lodata = datapkl["client_" + str(i_client)]
        n_samples = len(lodata)
        client_data_indices.append((start_idx, start_idx + n_samples))
        start_idx += n_samples
        
        # Add noise and cluster
        laplace_noise = np.random.laplace(loc=0.0, scale=1/epsilon, size=lodata.shape)
        lodata_noisy = lodata + laplace_noise
        
        cluster = KMeans(n_clusters=k1, random_state=42, n_init=10)
        centers = cluster.fit(lodata_noisy).cluster_centers_
        corepoints.append(centers)
    
    # Prepare server data and get true labels for all local centroids
    serverdata = np.concatenate(corepoints, axis=0)
    cnum = len(set(true_labels))
    
    # Get global centroids from SNN
    centroid_indices, assignment = SNN_optimized(k2, cnum, serverdata)
    
    # Find the true labels for the selected centroids
    centroid_labels = []
    for idx in centroid_indices:
        # Find which client this centroid came from
        client_idx = np.searchsorted(
            np.cumsum([len(c) for c in corepoints]), 
            idx, 
            side='right'
        ) - 1
        
        # Get the data range for this client
        client_start, client_end = client_data_indices[client_idx]
        client_true_labels = true_labels[client_start:client_end]
        
        # Get the label distribution for this centroid's cluster
        cluster_points = np.where(assignment == assignment[idx])[0]
        cluster_labels = []
        
        for pt_idx in cluster_points:
            pt_client_idx = np.searchsorted(
                np.cumsum([len(c) for c in corepoints]), 
                pt_idx, 
                side='right'
            ) - 1
            pt_start, pt_end = client_data_indices[pt_client_idx]
            pt_global_idx = pt_start + (pt_idx - sum(len(c) for c in corepoints[:pt_client_idx]))
            cluster_labels.append(true_labels[pt_global_idx])
        
        if cluster_labels:
            centroid_label = max(set(cluster_labels), key=cluster_labels.count)
            centroid_labels.append(centroid_label)
        else:
            centroid_label = -1  # No label found
            centroid_labels.append(centroid_label)
    
    # Analyze label distribution
    unique_labels, counts = np.unique(centroid_labels, return_counts=True)
    duplicate_info = [(label, count) for label, count in zip(unique_labels, counts) if count > 1]
    
    print("\nCentroid Label Analysis:")
    print(f"Selected centroids indices: {centroid_indices}")
    print(f"Corresponding true labels: {centroid_labels}")
    if duplicate_info:
        print(f"Duplicate labels found: {duplicate_info}")
    else:
        print("No duplicate labels among selected centroids")
    
    # Original clustering evaluation
    finalcenter = [serverdata[i] for i in centroid_indices]
    
    idx = []
    for point in data:
        distances = [np.linalg.norm(point - center) for center in finalcenter]
        idx.append(np.argmin(distances))
    
    arr = np.array(idx)
    ari = round(adjusted_rand_score(true_labels, arr), 4)
    nmi = round(normalized_mutual_info_score(true_labels, arr), 4)
    
    return ari, nmi, [k1, cnum, k2], centroid_labels


def run_single_experiment(args):
    """Single experiment runner for multiprocessing"""
    
    data_path, use_gpu, k1,k2,epsilon,seed = args
    return nnfc_optimized(data_path, use_gpu,k1,k2,epsilon,seed)




def run_experiments_parallel(data_path, n_runs, n_processes, use_gpu, k1, k2, epsilon, seed):
    """Run experiments in parallel across different random seeds"""
    # Generate random seeds for experiments
   
    args_list = [(data_path, seed, k1, k2, epsilon,seed)]
    
    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(run_single_experiment, args_list))
    
    return results





def evaluate_with_seeds(data_path, use_gpu=False, n_runs=1000, n_processes=None, k1=None, k2=None, seed=None, epsilon=None):
    """Evaluate performance across multiple seeds"""
    seed_results = {'ari_max': [], 'nmi_max': []}
    SEEDS = list(range(seed))
    
    for seed in SEEDS:
        print(f"\n--- Evaluating with seed={seed} ---")
        
        # Set global seed for this iteration
        np.random.seed(seed)
    
        
        results = run_experiments_parallel(data_path, n_runs, n_processes, use_gpu, k1, k2, epsilon, seed)
        
        # Handle None/NaN/NA values by converting them to 0
        ari_scores = [r[0] if r[0] is not None and not np.isnan(r[0]) else 0 for r in results]
        nmi_scores = [r[1] if r[1] is not None and not np.isnan(r[1]) else 0 for r in results]
        
        # Only consider positive scores (if you still want this condition)
        ari_scores = [score for score in ari_scores if score > 0]
        nmi_scores = [score for score in nmi_scores if score > 0]
        
        if ari_scores:
            seed_results['ari_max'].append(max(ari_scores))
        else:
            seed_results['ari_max'].append(0)
            
        if nmi_scores:
            seed_results['nmi_max'].append(max(nmi_scores))
        else:
            seed_results['nmi_max'].append(0)
    
    # Calculate statistics
    ari_mean = np.mean(seed_results['ari_max'])
    ari_std = np.std(seed_results['ari_max'])
    nmi_mean = np.mean(seed_results['nmi_max'])
    nmi_std = np.std(seed_results['nmi_max'])
    
    print("\n" + "="*50)
    print(f"SEED EVALUATION RESULTS ({len(SEEDS)} seeds)")
    print("="*50)
    print(f"ARI (max): {ari_mean:.4f} ± {ari_std:.4f}")
    print(f"NMI (max): {nmi_mean:.4f} ± {nmi_std:.4f}")
    
    return seed_results



def run_experiment(datanames, seeds, n_runs, use_gpu,save_path,config_file):
    """Updated main function"""

    config_path = config_file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # dataname = ['mnist2d']
    use_gpu = GPU_AVAILABLE
    n_processes = mp.cpu_count() - 1
    
    for dataset in datanames:
        data_path = f'dataset/{dataset}fed.pkl'
        print(f"\nProcessing dataset: {dataset}")
        
        if not os.path.exists(data_path):
            print(f"Dataset {data_path} not found, skipping...")
            continue


        if dataset not in config:
            print(f"Config for {dataset} not found, skipping...")
            continue

        k1 = config[dataset]['k1']
        k2 = config[dataset]['k2']
        epsilon = config['epsilon']
        
        # Evaluate with 10 seeds
        seed_results = evaluate_with_seeds(data_path, use_gpu, n_runs=n_runs, n_processes=n_processes,k1=k1,k2=k2,seed=seeds,epsilon=epsilon)
        

        # Save seed results
        with open(save_path, 'a') as f:
            f.write(f"{data_path}\n")
            f.write(f"ARI (max) across seeds: {seed_results['ari_max']}\n")
            f.write(f"NMI (max) across seeds: {seed_results['nmi_max']}\n")
            f.write(f"Mean ± Std - ARI: {np.mean(seed_results['ari_max']):.4f} ± {np.std(seed_results['ari_max']):.4f}\n")
            f.write(f"Mean ± Std - NMI: {np.mean(seed_results['nmi_max']):.4f} ± {np.std(seed_results['nmi_max']):.4f}\n\n")


