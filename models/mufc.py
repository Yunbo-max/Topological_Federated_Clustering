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
    # true_labels = dataset['true_label']
    datapkl = load_dataset(data_path)
    data = np.array(datapkl['full_data'])
    true_labels = np.array(datapkl['true_label'])  # Original labels for full data
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
    
    serverdata = np.concatenate(x_fed, axis=0)
    
    ###################
    # MUFC Algorithm
    ###################

    # For the centroids' true labels, we need to assign them based on their closest true data points
    centroid_true_labels = []
    for centroid in serverdata:
        # Find the closest original data point to this centroid
        closest_idx = np.argmin(np.linalg.norm(data - centroid, axis=1))
        centroid_true_labels.append(true_labels[closest_idx])
    centroid_true_labels = np.array(centroid_true_labels)

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

    # Visualization with true labels
    plot_clusters(serverdata, final_centroids, energy=None, true_labels=centroid_true_labels)

    
    print(f'Final centroids shape: {final_centroids.shape}')
    
    return final_centroids




def plot_clusters(data, synthetic_centroids, energy, true_labels=None, assignments=None):
    """Visualize clusters in 2D or higher dimensions using t-SNE when needed"""
    plt.figure(figsize=(8, 6))
    
    # Handle dimensionality
    d = data.shape[1]
    if d > 2:
        from sklearn.manifold import TSNE
        
        # Combine data and centroids for consistent t-SNE projection
        combined = np.vstack([data, synthetic_centroids])
        
        # Create t-SNE transformer (perplexity auto-adjusted)
        perplex = min(30, combined.shape[0]-1)
        tsne = TSNE(n_components=2, perplexity=perplex, random_state=42)
        projected = tsne.fit_transform(combined)
        
        # Split back into data and centroids
        data_2d = projected[:len(data)]
        centroids_2d = projected[len(data):]
        
        plot_title = "t-SNE Projection of "
    else:
        data_2d = data
        centroids_2d = synthetic_centroids
        plot_title = ""

    # Plot all points with true labels if available
    if true_labels is not None:
        unique_labels = np.unique(true_labels)
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        
        # Plot points with true labels
        for i, label in enumerate(unique_labels):
            mask = (true_labels == label)
            plt.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                        color=cmap(i), alpha=0.6, s=30,
                        label=f'True class {label}')
    else:
        # Only apply energy mask if the dimensions match
        if len(energy) == len(data_2d):
            # Calculate energy threshold
            min_e, max_e = np.min(energy), np.max(energy)
            threshold = min_e
            high_energy_mask = energy > threshold
            
            # Plot all points (below threshold in blue)
            plt.scatter(data_2d[~high_energy_mask, 0], data_2d[~high_energy_mask, 1], 
                        c='lightblue', alpha=0.4, s=30, label='Below energy threshold')
            
            # Plot high-energy points (red)
            plt.scatter(data_2d[high_energy_mask, 0], data_2d[high_energy_mask, 1], 
                        c='red', alpha=0.6, s=30, label='Above energy threshold')
        else:
            # If dimensions don't match, just plot all points
            plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                        c='blue', alpha=0.6, s=30, label='All points')
    
    # Plot cluster assignments if available and not all zeros
    if assignments is not None and not np.all(assignments == 0) and len(assignments) == len(data_2d):
        # Override colors for assigned points
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=assignments, 
                             cmap='tab20', alpha=0.7, s=30, label='Cluster assignments')
        plt.colorbar(scatter, label='Cluster ID')
    
    # Plot synthetic centroids (large gold stars)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                marker='*', s=400, c='gold', edgecolors='black',
                linewidths=2, label='Synthetic Centroids', zorder=5)
    
    # Add numbering to centroids
    for i, centroid in enumerate(centroids_2d):
        plt.annotate(f'{i+1}', (centroid[0], centroid[1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='black')
    
    title_suffix = "with True Labels" if true_labels is not None else ""
    # plt.title(f"{plot_title}Synthetic Centroids {title_suffix}")
    plt.xlabel('Component 1' if d > 2 else 'Feature 1')
    plt.ylabel('Component 2' if d > 2 else 'Feature 2')
    plt.legend(loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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

def run_experiments_parallel(data_path, n_runs, n_processes, use_gpu, k1, k2,epsilon,seed):
    """Run experiments in parallel across different random seeds"""
    # Generate random seeds for experiments
   
    args_list = [(data_path, seed, k1, k2, use_gpu,epsilon)]
    
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



