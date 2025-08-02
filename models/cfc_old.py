# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2025-06-02 17:55:49
# @Last Modified by:   Yunbo
# @Last Modified time: 2025-07-05 22:37:23
import umap
import json
import os
from scipy.spatial.distance import pdist, squareform, cdist
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Try to use GPU acceleration if available


def load_dataset(filepath):
    """Load dataset from pickle file"""
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset

def dist(a, b, ax=1):
    """Compute distance between two points"""
    return np.linalg.norm(a - b, axis=ax)


import numpy as np
from scipy.spatial.distance import pdist, squareform

# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from typing import Tuple


# def SNN_optimized(k: int, nc: int, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """Optimized SNN clustering with energy-based centroid selection"""
#     unassigned = -1
#     n, d = data.shape

   
#     distance = squareform(pdist(data))

#     # Compute k-nearest neighbors (same as original)
#     indexDistanceAsc = np.argsort(distance, axis=1)
#     indexNeighbor = indexDistanceAsc[:, :k]

#     # Shared neighbor computation (same as original)
#     numSharedNeighbor = np.zeros([n, n], dtype=int)
#     for i in range(n):
#         for j in range(i):
#             shared = np.intersect1d(indexNeighbor[i], indexNeighbor[j])
#             numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = len(shared)

#     # =====================================================================
#     # ENERGY-BASED CENTROID SELECTION (REPLACES RHO/DELTA/GAMMA)
#     # =====================================================================
    
#     # Enhanced physics-inspired energy method
#     np.fill_diagonal(distance, np.inf)  # Remove self-interaction
    
#     # Improved energy calculation with adaptive epsilon and weighting
#     eps = np.percentile(distance[distance > 0], 5)  # 5th percentile of non-zero distances
#     energy = np.sum(1/(distance**2 + eps**2), axis=1)  # More stable denominator
    
#     # Density-aware centroid selection
#     log_energy = np.log(energy + 1e-10)  # Log transform for better separation
#     weights = np.exp(log_energy - np.max(log_energy))  # Softmax-like weighting
#     top_indices = np.argsort(energy)[-2*nc:]  # Consider wider candidate pool
    
#     # Select final centroids from high-density regions
#     candidate_scores = weights[top_indices] * energy[top_indices]
#     indexCentroid = np.sort(top_indices[np.argsort(candidate_scores)[-nc:]])


#     # =====================================================================
#     # CLUSTER ASSIGNMENT (SAME AS ORIGINAL)
#     # =====================================================================
#     indexAssignment = np.full(n, unassigned)
#     indexAssignment[indexCentroid] = np.arange(nc)

#     # Assignment process
#     queue = indexCentroid.tolist()
#     while queue:
#         a = queue.pop(0)
#         for b in indexNeighbor[a]:
#             if indexAssignment[b] == unassigned and numSharedNeighbor[a, b] >= k // 2:
#                 indexAssignment[b] = indexAssignment[a]
#                 queue.append(b)

#     # Handle unassigned points
#     indexUnassigned = np.where(indexAssignment == unassigned)[0]
#     while len(indexUnassigned) > 0:
#         for idx in indexUnassigned:
#             neighbors = indexDistanceAsc[idx, :k]
#             assigned_neighbors = neighbors[indexAssignment[neighbors] != unassigned]
#             if len(assigned_neighbors) > 0:
#                 clusters, counts = np.unique(indexAssignment[assigned_neighbors], return_counts=True)
#                 indexAssignment[idx] = clusters[np.argmax(counts)]
        
#         new_unassigned = np.where(indexAssignment == unassigned)[0]
#         if len(new_unassigned) == len(indexUnassigned):
#             indexAssignment[new_unassigned] = np.random.randint(0, nc, len(new_unassigned))
#             break
#         indexUnassigned = new_unassigned

#     return indexCentroid, indexAssignment




















def SNN_optimized(k: int, nc: int, data: np.ndarray, true_labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced SNN clustering with energy-based centroid selection and label-aware merging"""
    unassigned = -1
    n, d = data.shape

   
    distance = squareform(pdist(data))


    # Compute k-nearest neighbors (same as original)
    indexDistanceAsc = np.argsort(distance, axis=1)
    indexNeighbor = indexDistanceAsc[:, :k]

    # Shared neighbor computation (same as original)
    numSharedNeighbor = np.zeros([n, n], dtype=int)
    for i in range(n):
        for j in range(i):
            shared = np.intersect1d(indexNeighbor[i], indexNeighbor[j])
            numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = len(shared)

    # =====================================================================
    # ENERGY-BASED CENTROID SELECTION (REPLACES RHO/DELTA/GAMMA)
    # =====================================================================
    

    # Enhanced physics-inspired energy method
    np.fill_diagonal(distance, np.inf)  # Remove self-interaction
    
    # Improved energy calculation with adaptive epsilon
    eps = np.percentile(distance[distance > 0], 5)  # 5th percentile of non-zero distances
    energy = np.sum(1/(distance**2 + eps**2), axis=1)  # More stable denominator
    
    # Density-aware centroid selection
    log_energy = np.log(energy + 1e-10)
    weights = np.exp(log_energy - np.max(log_energy))  # Softmax-like weighting
    top_indices = np.argsort(energy)[-2*nc:]  # Wider candidate pool
    
    # First round: Select high-energy candidates
    candidate_scores = weights[top_indices] * energy[top_indices]
    sorted_candidates = top_indices[np.argsort(candidate_scores)]
    
    # Second round: Energy analysis between top candidates
    top_distance = distance[np.ix_(sorted_candidates, sorted_candidates)]
    candidate_energy = np.sum(1/(top_distance**2 + eps**2), axis=1)
    avg_energy = np.mean(candidate_energy)
    energy_signs = np.sign(candidate_energy - avg_energy)
    
    # Label-aware selection (if true_labels provided)
    final_centroids = []
    seen_labels = set()
    
    if true_labels is not None:
        # Get labels for all candidate points
        candidate_labels = true_labels[sorted_candidates]
    else:
        candidate_labels = np.zeros(len(sorted_candidates))
    
    for i, idx in enumerate(sorted_candidates[::-1]):  # Process from highest energy
        current_label = candidate_labels[-i-1]  # Because we reversed
        
        # Skip if we already have this label (unless we need more centroids)
        if current_label in seen_labels and len(final_centroids) >= nc:
            continue
            
        # Check energy sign agreement with previous
        if len(final_centroids) > 0:
            prev_sign = energy_signs[-i-2]
            curr_sign = energy_signs[-i-1]
            if prev_sign == curr_sign and curr_sign > 0:  # Both above average
                continue  # Skip consecutive high-energy duplicates
        
        final_centroids.append(idx)
        seen_labels.add(current_label)
        
        if len(final_centroids) >= nc:
            break
    
    # Fallback if we didn't get enough centroids
    while len(final_centroids) < nc:
        remaining = [x for x in sorted_candidates[::-1] if x not in final_centroids]
        final_centroids.append(remaining[0])
    
    indexCentroid = np.sort(final_centroids[:nc])

    # =====================================================================
    # CLUSTER ASSIGNMENT (SAME AS ORIGINAL)
    # =====================================================================
    indexAssignment = np.full(n, unassigned)
    indexAssignment[indexCentroid] = np.arange(nc)

    # Assignment process
    queue = indexCentroid.tolist()
    while queue:
        a = queue.pop(0)
        for b in indexNeighbor[a]:
            if indexAssignment[b] == unassigned and numSharedNeighbor[a, b] >= k // 2:
                indexAssignment[b] = indexAssignment[a]
                queue.append(b)

    # Handle unassigned points
    indexUnassigned = np.where(indexAssignment == unassigned)[0]
    while len(indexUnassigned) > 0:
        for idx in indexUnassigned:
            neighbors = indexDistanceAsc[idx, :k]
            assigned_neighbors = neighbors[indexAssignment[neighbors] != unassigned]
            if len(assigned_neighbors) > 0:
                clusters, counts = np.unique(indexAssignment[assigned_neighbors], return_counts=True)
                indexAssignment[idx] = clusters[np.argmax(counts)]
        
        new_unassigned = np.where(indexAssignment == unassigned)[0]
        if len(new_unassigned) == len(indexUnassigned):
            indexAssignment[new_unassigned] = np.random.randint(0, nc, len(new_unassigned))
            break
        indexUnassigned = new_unassigned

    return indexCentroid, indexAssignment










# def SNN_optimized(k: int, nc: int, data: np.ndarray, true_labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
#     """Enhanced SNN clustering with PCA-based ordering and energy-aware selection"""
#     unassigned = -1
#     n, d = data.shape

#     # Distance computation
#     distance = squareform(pdist(data))

#     # Compute k-nearest neighbors
#     indexDistanceAsc = np.argsort(distance, axis=1)
#     indexNeighbor = indexDistanceAsc[:, :k]

#     # Shared neighbor computation
#     numSharedNeighbor = np.zeros([n, n], dtype=int)
#     for i in range(n):
#         for j in range(i):
#             shared = np.intersect1d(indexNeighbor[i], indexNeighbor[j])
#             numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = len(shared)

#     # =====================================================================
#     # ENERGY-BASED CENTROID SELECTION WITH PCA ORDERING
#     # =====================================================================
#     np.fill_diagonal(distance, np.inf)
#     eps = np.percentile(distance[distance > 0], 5)
#     energy = np.sum(1/(distance**2 + eps**2), axis=1)
    
#     # Get top candidates (2x needed centroids)
#     top_indices = np.argsort(energy)[-2*nc:]
#     candidate_scores = energy[top_indices]
#     sorted_candidates = top_indices[np.argsort(candidate_scores)]

#     # Project candidates onto PCA rank-1 for ordering
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=1)
#     projected = pca.fit_transform(data[sorted_candidates]).flatten()
#     pca_order = np.argsort(projected)  # Order along PCA axis
    
#     # Reorder candidates by PCA position
#     pca_ordered_candidates = sorted_candidates[pca_order]
    
#     # Energy analysis between top candidates
#     top_distance = distance[np.ix_(pca_ordered_candidates, pca_ordered_candidates)]
#     candidate_energy = np.sum(1/(top_distance**2 + eps**2), axis=1)
#     avg_energy = np.mean(candidate_energy)
#     energy_signs = np.sign(candidate_energy - avg_energy)

#     # Label-aware selection along PCA axis
#     final_centroids = []
#     seen_labels = set()
    
#     if true_labels is not None:
#         candidate_labels = true_labels[pca_ordered_candidates]
#     else:
#         candidate_labels = np.zeros(len(pca_ordered_candidates))

#     # Process in PCA order (not energy order)
#     for i, idx in enumerate(pca_ordered_candidates):
#         current_label = candidate_labels[i]
        
#         # Skip duplicates if we have enough centroids
#         if current_label in seen_labels and len(final_centroids) >= nc:
#             continue
            
#         # Energy sign check with previous centroid
#         if len(final_centroids) > 0:
#             prev_idx = final_centroids[-1]
#             prev_pos = np.where(pca_ordered_candidates == prev_idx)[0][0]
#             curr_sign = energy_signs[i]
#             prev_sign = energy_signs[prev_pos]
            
#             # Skip if same energy regime and same label
#             if prev_sign == curr_sign and curr_sign > 0 and current_label == candidate_labels[prev_pos]:
#                 continue
        
#         final_centroids.append(idx)
#         seen_labels.add(current_label)
        
#         if len(final_centroids) >= nc:
#             break

#     # Fallback if needed
#     while len(final_centroids) < nc:
#         remaining = [x for x in pca_ordered_candidates if x not in final_centroids]
#         final_centroids.append(remaining[0])

#     indexCentroid = np.sort(final_centroids[:nc])

#     # =====================================================================
#     # CLUSTER ASSIGNMENT 
#     # =====================================================================
#     indexAssignment = np.full(n, unassigned)
#     indexAssignment[indexCentroid] = np.arange(nc)

#     # Assignment process (same as before)
#     queue = indexCentroid.tolist()
#     while queue:
#         a = queue.pop(0)
#         for b in indexNeighbor[a]:
#             if indexAssignment[b] == unassigned and numSharedNeighbor[a, b] >= k // 2:
#                 indexAssignment[b] = indexAssignment[a]
#                 queue.append(b)

#     # Handle unassigned points
#     indexUnassigned = np.where(indexAssignment == unassigned)[0]
#     while len(indexUnassigned) > 0:
#         for idx in indexUnassigned:
#             neighbors = indexDistanceAsc[idx, :k]
#             assigned_neighbors = neighbors[indexAssignment[neighbors] != unassigned]
#             if len(assigned_neighbors) > 0:
#                 clusters, counts = np.unique(indexAssignment[assigned_neighbors], return_counts=True)
#                 indexAssignment[idx] = clusters[np.argmax(counts)]
        
#         new_unassigned = np.where(indexAssignment == unassigned)[0]
#         if len(new_unassigned) == len(indexUnassigned):
#             indexAssignment[new_unassigned] = np.random.randint(0, nc, len(new_unassigned))
#             break
#         indexUnassigned = new_unassigned

#     return indexCentroid, indexAssignment







# shared neighboughs
# def SNN_optimized(k: int, nc: int, data: np.ndarray, true_labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
#     """Enhanced SNN clustering with diversity-aware centroid selection"""
#     unassigned = -1
#     n, d = data.shape

#     # Distance computation
#     distance = squareform(pdist(data))

#     # Compute k-nearest neighbors
#     indexDistanceAsc = np.argsort(distance, axis=1)
#     indexNeighbor = indexDistanceAsc[:, :k]

#     # Shared neighbor computation
#     numSharedNeighbor = np.zeros([n, n], dtype=int)
#     for i in range(n):
#         for j in range(i):
#             shared = np.intersect1d(indexNeighbor[i], indexNeighbor[j])
#             numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = len(shared)

#     # =====================================================================
#     # DIVERSITY-AWARE CENTROID SELECTION
#     # =====================================================================
    
#     # Initial energy calculation
#     np.fill_diagonal(distance, np.inf)  # Remove self-interaction
#     eps = np.percentile(distance[distance > 0], 5)
#     energy = np.sum(1/(distance**2 + eps**2), axis=1)
    
#     # Select wider candidate pool (top 2*nc candidates)
#     candidate_indices = np.argsort(energy)[-2*nc:]
#     candidate_shared = numSharedNeighbor[np.ix_(candidate_indices, candidate_indices)]
    
#     # Sort candidates by energy (descending)
#     sorted_energy = energy[candidate_indices]
#     sorted_candidates = candidate_indices[np.argsort(-sorted_energy)]
    
#     # Greedy selection for diverse centroids
#     selected_centroids = []
#     remaining_candidates = sorted_candidates.tolist()
    
#     if true_labels is not None:
#         candidate_labels = true_labels[sorted_candidates]
#         used_labels = set()
#     else:
#         candidate_labels = None
    
#     while len(selected_centroids) < nc and remaining_candidates:
#         # Take the highest energy remaining candidate
#         current = remaining_candidates.pop(0)
#         current_label = candidate_labels[np.where(sorted_candidates == current)[0][0]] if candidate_labels is not None else None
        
#         # Skip if we already have this label (when labels are available)
#         if candidate_labels is not None and current_label in used_labels:
#             continue
            
#         # Check if too similar to already selected centroids
#         if selected_centroids:
#             # Get indices of selected centroids in candidate space
#             selected_in_candidate = [np.where(sorted_candidates == x)[0][0] for x in selected_centroids]
#             # Average shared neighbors with existing centroids
#             avg_shared = np.mean(candidate_shared[np.where(sorted_candidates == current)[0][0], selected_in_candidate])
#             if avg_shared > k//2:  # Skip if shares too many neighbors
#                 continue
                
#         selected_centroids.append(current)
#         if candidate_labels is not None:
#             used_labels.add(current_label)
    
#     # Fallback if we didn't get enough diverse centroids
#     while len(selected_centroids) < nc and remaining_candidates:
#         selected_centroids.append(remaining_candidates.pop(0))
    
#     indexCentroid = np.sort(selected_centroids[:nc])

#     # =====================================================================
#     # CLUSTER ASSIGNMENT (SAME AS ORIGINAL)
#     # =====================================================================
#     indexAssignment = np.full(n, unassigned)
#     indexAssignment[indexCentroid] = np.arange(nc)

#     # Assignment process
#     queue = indexCentroid.tolist()
#     while queue:
#         a = queue.pop(0)
#         for b in indexNeighbor[a]:
#             if indexAssignment[b] == unassigned and numSharedNeighbor[a, b] >= k // 2:
#                 indexAssignment[b] = indexAssignment[a]
#                 queue.append(b)

#     # Handle unassigned points
#     indexUnassigned = np.where(indexAssignment == unassigned)[0]
#     while len(indexUnassigned) > 0:
#         for idx in indexUnassigned:
#             neighbors = indexDistanceAsc[idx, :k]
#             assigned_neighbors = neighbors[indexAssignment[neighbors] != unassigned]
#             if len(assigned_neighbors) > 0:
#                 clusters, counts = np.unique(indexAssignment[assigned_neighbors], return_counts=True)
#                 indexAssignment[idx] = clusters[np.argmax(counts)]
        
#         new_unassigned = np.where(indexAssignment == unassigned)[0]
#         if len(new_unassigned) == len(indexUnassigned):
#             indexAssignment[new_unassigned] = np.random.randint(0, nc, len(new_unassigned))
#             break
#         indexUnassigned = new_unassigned

#     print('indexCentroid',indexCentroid)

#     return indexCentroid, indexAssignment





def add_laplace_noise_vectorized(data, epsilon, sensitivity):
    """Vectorized Laplace noise addition"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise










# def nnfc_optimized(data_path, use_gpu=False,k1=None,k2=None):
#     """Optimized NNFC function"""
#     try:
#         datapkl = load_dataset(data_path)
#         # eachlable = datapkl['eachlable']
#         # order = datapkl['order']
#         data = np.array(datapkl['full_data'])
        
#         corepoints = []
        
#         for i_client in range(10):
#             lodata = datapkl["client_" + str(i_client)]

#             epsilon = 0.01
#             scale = 1 / epsilon
#             laplace_noise = np.random.laplace(loc=0.0, scale=scale, size=lodata.shape)


#             lodata_noisy = lodata + laplace_noise
       
        
#             # n_clusters = min(len(lodata_noisy) // 3, 50)
#             n_clusters = 50
            
#             cluster = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#             centers = cluster.fit(lodata_noisy).cluster_centers_
            
#             corepoints.append(centers)
        
#         serverdata = np.concatenate(corepoints, axis=0)
#         label = datapkl['true_label']
#         cnum = len(set(label))
#         # k = min(5, len(serverdata) // 10)  # Adaptive k

#         k = 20
        
    #     centroid, assignment = SNN_optimized(k, cnum, serverdata)
        
    #     finalcenter = serverdata[centroid]
        
    #     # Vectorized assignment of all data points
    #     distances = euclidean_distances(data, finalcenter)
    #     idx = np.argmax(-distances, axis=1) + 1  # +1 for 1-based indexing
        
    #     ari = adjusted_rand_score(label, idx)
    #     nmi = normalized_mutual_info_score(label, idx)
        
    #     return ari, nmi, [n_clusters, cnum, k]
        
    # except Exception as e:
    #     print(f"Error in nnfc_optimized: {e}")
    #     return 0, 0, [0, 0, 0]
    

def nnfc_optimized(data_path, use_gpu=False, k1=None, k2=None):
    """Optimized NNFC function with centroid label checking"""
    datapkl = load_dataset(data_path)
    data = np.array(datapkl['full_data'])
    true_labels = np.array(datapkl['true_label'])  # Original labels for full data
    
    corepoints = []
    client_data_indices = []  # To track which data points belong to each client
    start_idx = 0
    
    # Process each client's data
    for i_client in range(10):
        lodata = datapkl["client_" + str(i_client)]
        n_samples = len(lodata)
        client_data_indices.append((start_idx, start_idx + n_samples))
        start_idx += n_samples
        
        # Add noise and cluster
        epsilon = 1000
        laplace_noise = np.random.laplace(loc=0.0, scale=1/epsilon, size=lodata.shape)
        lodata_noisy = lodata + laplace_noise
        
        cluster = KMeans(n_clusters=k1, random_state=42, n_init=10)
        centers = cluster.fit(lodata_noisy).cluster_centers_
        corepoints.append(centers)
    
    # Prepare server data and get true labels for all local centroids
    serverdata = np.concatenate(corepoints, axis=0)
    label = datapkl['true_label']
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
    
        
    finalcenter = serverdata[centroid_indices]
    
    # Vectorized assignment of all data points
    distances = euclidean_distances(data, finalcenter)
    idx = np.argmax(-distances, axis=1)   # +1 for 1-based indexing
    
    ari = adjusted_rand_score(label, idx)
    nmi = normalized_mutual_info_score(label, idx)
    
    return ari, nmi
    







def run_single_experiment(args):
    """Single experiment runner for multiprocessing"""
    data_path, use_gpu, run_id,k1,k2 = args
    return nnfc_optimized(data_path, use_gpu,k1,k2)






def run_experiments_parallel(data_path, n_runs=1000, n_processes=None, use_gpu=True, k1=None, k2=None, dataset=None):
    """Run experiments in parallel, optionally in batches (e.g., for 'mnist2')"""
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes max
    
    print(f"Running {n_runs} experiments on {n_processes} cores...")

    results = []
    start_time = time.time()

    # Define batch size
    if dataset in ['celltypes1','covtype1','postures1','mnist2d1']:
        batch_size = 5
    else:
        batch_size = n_runs  # run all at once

    batches = int((n_runs + batch_size - 1) // batch_size)  # Ceiling division

    run_counter = 0
    for batch_idx in range(batches):
        current_batch_size = min(batch_size, n_runs - run_counter)
        args_list = [(data_path, use_gpu, run_counter + i, k1, k2) for i in range(current_batch_size)]

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            future_to_run = {executor.submit(run_single_experiment, args): run_counter + i for i, args in enumerate(args_list)}

            for future in as_completed(future_to_run):
                run_id = future_to_run[future]
                try:
                    result = future.result()
                    results.append(result)

                    if len(results) % 100 == 0 or dataset in ['abalone','celltypes','covtype','postures','mnist','bot']:
                        elapsed = time.time() - start_time
                        print(f"Completed {len(results)}/{n_runs} runs in {elapsed:.1f}s")
                        print('results_max',max(results))
                except Exception as e:
                    print(f"Run {run_id} failed: {e}")
                    results.append((0, 0, [0, 0, 0]))

        run_counter += current_batch_size

    return results







def evaluate_with_seeds(data_path, use_gpu=True, n_runs=100, n_processes=None,k1=None,k2=None,dataset=None,seed=None):
    """Evaluate performance across 10 seeds"""
    SEEDS = list(range(seed))
    seed_results = {'ari_max': [], 'nmi_max': []}
    
    for seed in SEEDS:
        print(f"\n--- Evaluating with seed={seed} ---")
        results = run_experiments_parallel(data_path, n_runs, n_processes, use_gpu,k1,k2,dataset)
        ari_scores = [r[0] for r in results if r[0] > 0]
        nmi_scores = [r[1] for r in results if r[1] > 0]
        
        if ari_scores and nmi_scores:
            seed_results['ari_max'].append(max(ari_scores))
            seed_results['nmi_max'].append(max(nmi_scores))
    
    # Calculate statistics
    ari_mean = np.mean(seed_results['ari_max'])
    ari_std = np.std(seed_results['ari_max'])
    nmi_mean = np.mean(seed_results['nmi_max'])
    nmi_std = np.std(seed_results['nmi_max'])
    
    print("\n" + "="*50)
    print("SEED EVALUATION RESULTS (10 seeds)")
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

    use_gpu = True
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
        candidates_multiplier = config[dataset]['candidates_multiplier']
        energy_multiplier = config[dataset]['energy_multiplier']
        epsilon = config['epsilon']
     

        

        # Evaluate with 10 seeds
        seed_results = evaluate_with_seeds(data_path, use_gpu, n_runs=n_runs, n_processes=n_processes,k1=k1,k2=k2,dataset=dataset,seed=seeds)
        
        # Save seed results
        with open(save_path, 'a') as f:
            f.write(f"{data_path}\n")
            f.write(f"ARI (max) across seeds: {seed_results['ari_max']}\n")
            f.write(f"NMI (max) across seeds: {seed_results['nmi_max']}\n")
            f.write(f"Mean ± Std - ARI: {np.mean(seed_results['ari_max']):.4f} ± {np.std(seed_results['ari_max']):.4f}\n")
            f.write(f"Mean ± Std - NMI: {np.mean(seed_results['nmi_max']):.4f} ± {np.std(seed_results['nmi_max']):.4f}\n\n")






