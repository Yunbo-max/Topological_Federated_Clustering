# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2025-07-02 19:46:26
# @Last Modified by:   Yunbo
# @Last Modified time: 2025-07-04 13:21:56
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############################
# MUFC Clustering Evaluation with Seeds
#############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import euclidean_distances
import argparse
import time
from tqdm import tqdm
import copy
import os
import os.path as osp
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json
import os

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Seeds for evaluation
SEEDS = list(range(100))

class MyKmeans:
    """Simplified MyKmeans class - replace with your actual implementation"""
    def __init__(self, k, max_iters=300):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.data = None
        self.n = 0
        
    def run(self, data):
        """Run K-means clustering"""
        self.data = data
        self.n = data.shape[0]
        
        # Use sklearn for now - replace with your implementation
        kmeans = KMeans(n_clusters=self.k, max_iter=self.max_iters, n_init=10, random_state=42)
        assignments = kmeans.fit_predict(data)
        self.centroids = kmeans.cluster_centers_
        
        return self.centroids, assignments, kmeans.inertia_
    
    def run_kpp_only(self, data):
        """Run only K-means++ initialization"""
        return self.run(data)  # Simplified - replace with your actual implementation
    
    def quantize_centroids(self, eps):
        """Quantize centroids"""
        if self.centroids is None:
            return None
        # Simple quantization - replace with your actual implementation
        return np.round(self.centroids / eps) * eps

def load_dataset(data_path):
    """Load federated dataset from pickle file"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def sample_points_in_bin(centroid, n_samples, eps):
    """Sample points within quantization bin"""
    # Simple uniform sampling within the bin
    dim = len(centroid)
    samples = np.random.uniform(-eps/2, eps/2, (n_samples, dim))
    return samples + centroid.reshape(1, -1)

def run_mufc_clustering(data_path, k1, k2, seed, num_clients=10, client_oversample=1, max_iters=300,epsilon=None):
    """
    Run MUFC clustering and return final centroids
    
    Args:
        data_path: Path to dataset
        k1: Number of clusters per client (dev_k)
        k2: Not used in MUFC (kept for compatibility)
        seed: Random seed
        num_clients: Number of federated clients
        client_oversample: Oversampling factor
        max_iters: Maximum iterations
    
    Returns:
        final_centroids: Cluster centers from MUFC
    """
    # Set random seed
    np.random.seed(seed)
    
    # Load dataset
    dataset = load_dataset(data_path)
    
    # Get true number of clusters from labels
    true_labels = dataset['true_label']
    k_true = len(np.unique(true_labels))
    
    # Fixed quantization epsilon
    quant_eps = 1 / np.sqrt(dataset["full_data"].shape[0])
    
    print(f'Running MUFC with {num_clients} clients, k1={k1}, k_true={k_true}')
    
    # Prepare federated data with differential privacy noise
    scale = 1 / epsilon
    x_fed = []
    
    for i_client in range(num_clients):
        client_data = dataset[f"client_{i_client}"]
        
        # Add Laplace noise for differential privacy
        laplace_noise = np.random.laplace(loc=0.0, scale=scale, size=client_data.shape)
        client_data_noisy = client_data + laplace_noise
        x_fed.append(client_data_noisy)
    
    ###################
    # MUFC Algorithm
    ###################
    
    # Client-side computation
    client_worst_time = 0
    kmeans_clients = []
    data_server = []
    
    print('Processing clients with MUFC...')
    for i_client in range(num_clients):
        start = time.time()
        
        # Use k1 as the number of clusters per client
        client_kmeans = MyKmeans(k=int(client_oversample * k1), max_iters=max_iters)
        
        # Run K-means on noisy client data
        _, client_assignments, _ = client_kmeans.run(x_fed[i_client])
        
        # Quantize client centroids
        client_quant_centroids = client_kmeans.quantize_centroids(quant_eps)
        
        client_time_used = time.time() - start
        client_worst_time = max(client_worst_time, client_time_used)
        
        # Generate server data from quantized centroids
        for i_point in range(client_kmeans.n):
            rnd_samples = sample_points_in_bin(
                client_quant_centroids[client_assignments[i_point]], 1, quant_eps
            )
            data_server.append(rnd_samples)
        
        # Store client information
        kmeans_clients.append(client_kmeans)
    
    # Server-side computation
    print('Running server-side clustering...')
    data_server = np.concatenate(data_server, axis=0)
    
    # Run K-means on server with k_true clusters
    kmeans_server = KMeans(
        n_clusters=k_true,
        max_iter=max_iters,
        n_init=10,
        random_state=seed
    ).fit(data_server)
    
    final_centroids = kmeans_server.cluster_centers_
    
    print(f'Final centroids shape: {final_centroids.shape}')
    
    return final_centroids

def run_single_experiment(args):
    """Run a single MUFC experiment with given parameters"""
    data_path, seed, k1, k2, use_gpu,epsilon = args
    
    try:
        # Load dataset
        dataset = load_dataset(data_path)
        
        # Extract full data and true labels
        data = np.array(dataset['full_data'])
        label = dataset['true_label']
        k_true = len(np.unique(label))
        
        print(f"Data shape: {data.shape}, True clusters: {k_true}")
        
        # Run MUFC clustering to get final centroids
        final_centers = run_mufc_clustering(data_path, k1, k2, seed,num_clients=10, client_oversample=1, max_iters=300,epsilon=epsilon)
        
        print(f'Final centers shape: {final_centers.shape}')
        print(f'Full data shape: {data.shape}')
        
        # Calculate distances and assignments
        distances = euclidean_distances(data, final_centers)
        print('Calculating final assignments')
        
        # Get cluster assignments (0-based indexing first)
        idx_0based = np.argmin(distances, axis=1)  # Find closest centroid
        idx = idx_0based + 1  # Convert to 1-based indexing
        
        print('Calculating ARI and NMI')
        # Calculate metrics
        ari = adjusted_rand_score(label, idx)
        nmi = normalized_mutual_info_score(label, idx)
        
        print(f'ARI: {ari:.4f}, NMI: {nmi:.4f}')
        
        return ari, nmi
    
    except Exception as e:
        print(f"Error in experiment with seed {seed}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0

def run_experiments_parallel(data_path, n_runs, n_processes, use_gpu, k1, k2,epsilon):
    """Run experiments in parallel across different random seeds"""
    # Generate random seeds for experiments
    seeds = np.random.randint(0, 10000, n_runs)
    
    # Prepare arguments for parallel processing
    args_list = [(data_path, seed, k1, k2, use_gpu,epsilon) for seed in seeds]
    
    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(run_single_experiment, args_list))
    
    return results



def evaluate_with_seeds(data_path, use_gpu=True, n_runs=100, n_processes=None,k1=None,k2=None,dataset=None,seed=None,epsilon=None):
    """Evaluate performance across 10 seeds"""
    SEEDS = list(range(seed))

    
    seed_results = {'ari_max': [], 'nmi_max': []}
    
    for seed in SEEDS:
        print(f"\n--- Evaluating with seed={seed} ---")
        results = run_experiments_parallel(data_path, n_runs, n_processes, use_gpu,k1,k2,epsilon)
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
        seed_results = evaluate_with_seeds(data_path, use_gpu, n_runs=n_runs, n_processes=n_processes,k1=k1,k2=k2,dataset=dataset,seed=seeds,epsilon=epsilon)
        

        # Save seed results
        with open(save_path, 'a') as f:
            f.write(f"{data_path}\n")
            f.write(f"ARI (max) across seeds: {seed_results['ari_max']}\n")
            f.write(f"NMI (max) across seeds: {seed_results['nmi_max']}\n")
            f.write(f"Mean ± Std - ARI: {np.mean(seed_results['ari_max']):.4f} ± {np.std(seed_results['ari_max']):.4f}\n")
            f.write(f"Mean ± Std - NMI: {np.mean(seed_results['nmi_max']):.4f} ± {np.std(seed_results['nmi_max']):.4f}\n\n")



