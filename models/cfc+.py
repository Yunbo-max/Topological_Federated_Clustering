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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform, pdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from typing import Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.ndimage import label
import numpy as np
import matplotlib.pyplot as plt



def generate_synthetic_candidates(data, n_candidates_multiplier=1.5):
    """Generate synthetic candidate points using a grid approach"""
    n_candidates = int(len(data) * n_candidates_multiplier)
    
    # Create grid based on data bounds
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    
    # Expand bounds slightly
    range_expand = 0.1
    mins -= range_expand * (maxs - mins)
    maxs += range_expand * (maxs - mins)
    
    # Generate grid points
    if data.shape[1] == 2:
        grid_size = int(np.sqrt(n_candidates))
        x = np.linspace(mins[0], maxs[0], grid_size)
        y = np.linspace(mins[1], maxs[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
    else:
        # For higher dimensions, use random sampling within bounds
        candidates = np.random.uniform(mins, maxs, (n_candidates, data.shape[1]))
    
    return candidates




def generate_synthetic_candidates(data, n_candidates_multiplier=2):
    """Generate synthetic candidate points for centroid selection"""
    n_points = len(data)
    n_candidates = n_points * n_candidates_multiplier
    
    # Get data bounds
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Generate random candidates within data bounds
    n_dims = data.shape[1]
    candidates = np.random.uniform(
        low=min_vals, 
        high=max_vals, 
        size=(n_candidates, n_dims)
    )
    
    return candidates

def kmeans_plus_plus_selection(candidates, candidate_energy, n_centroids):
    """Apply k-means++ initialization logic for centroid selection"""
    if len(candidates) == 0 or n_centroids == 0:
        return np.array([]).reshape(0, candidates.shape[1] if len(candidates) > 0 else 0)
    
    if n_centroids >= len(candidates):
        return candidates
    
    centroids = []
    
    # Step 1: Choose first centroid with probability proportional to energy
    energy_probs = candidate_energy / np.sum(candidate_energy)
    first_idx = np.random.choice(len(candidates), p=energy_probs)
    centroids.append(candidates[first_idx])
    
    # Step 2: Choose remaining centroids
    for _ in range(n_centroids - 1):
        # Calculate distances from each candidate to closest existing centroid
        distances_to_centroids = pairwise_distances(candidates, np.array(centroids))
        min_distances = np.min(distances_to_centroids, axis=1)
        
        # Combine distance and energy for selection probability
        # Higher energy and larger distance from existing centroids = higher probability
        combined_score = min_distances * candidate_energy
        
        # Avoid division by zero
        if np.sum(combined_score) == 0:
            probs = np.ones(len(candidates)) / len(candidates)
        else:
            probs = combined_score / np.sum(combined_score)
        
        # Choose next centroid
        next_idx = np.random.choice(len(candidates), p=probs)
        centroids.append(candidates[next_idx])
    
    return np.array(centroids)

def find_natural_centroids(data, nc, synthetic_candidates, candidate_energy,energy_threshold):
    """Identify natural centroids using k-means++ initialization with energy weighting"""
    
    # Handle edge cases
    if len(synthetic_candidates) == 0:
        print("Warning: No synthetic candidates generated")
        return np.array([]).reshape(0, data.shape[1])
    
    if len(candidate_energy) == 0:
        print("Warning: No candidate energies calculated")
        return np.array([]).reshape(0, data.shape[1])
    
    # 1. Threshold to identify energy islands
    if len(candidate_energy) < 2:
        print("Warning: Insufficient candidate energies for percentile calculation")
        return synthetic_candidates[:min(nc, len(synthetic_candidates))]
    
    energy_threshold = np.percentile(candidate_energy, energy_threshold)  # Top 15% as islands
    island_mask = candidate_energy > energy_threshold
    
    if not np.any(island_mask):
        print("Warning: No energy islands found, returning highest energy points")
        top_indices = np.argsort(candidate_energy)[-nc:]
        return synthetic_candidates[top_indices]
    
    # 2. Get island candidates
    island_candidates = synthetic_candidates[island_mask]
    island_energies = candidate_energy[island_mask]
    
    # 3. Use k-means++ initialization on island candidates
    if len(island_candidates) < nc:
        print(f"Warning: Only {len(island_candidates)} island candidates for {nc} centroids")
        # Use all island candidates and fill remaining with k-means++ from all candidates
        remaining_nc = nc - len(island_candidates)
        if remaining_nc > 0:
            # Apply k-means++ to all candidates for remaining centroids
            additional_centroids = kmeans_plus_plus_selection(
                synthetic_candidates, candidate_energy, remaining_nc
            )
            if len(additional_centroids) > 0:
                all_centroids = np.vstack([island_candidates, additional_centroids])
            else:
                all_centroids = island_candidates
        else:
            all_centroids = island_candidates
    else:
        # Apply k-means++ to island candidates
        all_centroids = kmeans_plus_plus_selection(island_candidates, island_energies, nc)
    
    # 4. Final refinement: If we have more centroids than requested, select top by energy
    if len(all_centroids) > nc:
        # Find energy for each centroid
        centroid_energies = []
        for centroid in all_centroids:
            # Find closest candidate point
            distances = np.sum((synthetic_candidates - centroid)**2, axis=1)
            closest_idx = np.argmin(distances)
            centroid_energies.append(candidate_energy[closest_idx])
        
        top_indices = np.argsort(centroid_energies)[-nc:]
        return all_centroids[top_indices]
    
    return all_centroids

def penalized_energy_centroids(nc ,data,candidates_multiplier=None, energy_multiplier=None,energy_threshold=None):
    """Improved centroid selection using k-means++ initialization"""
    
    # Input validation
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be None or empty")
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    if data.shape[0] < nc:
        print(f"Warning: Requested {nc} centroids but only {data.shape[0]} data points available")
        nc = data.shape[0]
    
    # Generate synthetic candidates
    try:
        synthetic_candidates = generate_synthetic_candidates(data, n_candidates_multiplier=candidates_multiplier)
    except Exception as e:
        print(f"Error generating synthetic candidates: {e}")
        # Fallback: use actual data points
        synthetic_candidates = data.copy()
    
    # Calculate candidate energies
    try:
        candidate_distances = pairwise_distances(data, synthetic_candidates)
        eps = 1e-6
        candidate_energy = np.sum(1/(candidate_distances**energy_multiplier + eps**1), axis=0)
    except Exception as e:
        print(f"Error calculating candidate energies: {e}")
        # Fallback: uniform energy
        candidate_energy = np.ones(len(synthetic_candidates))
    
    # Find natural centroids using k-means++ approach
    try:
        centroids = find_natural_centroids(data, nc, synthetic_candidates, candidate_energy,energy_threshold)
    except Exception as e:
        print(f"Error finding natural centroids: {e}")
        # Fallback: standard k-means++ on original data
        if len(data) >= nc:
            kmeans = KMeans(n_clusters=nc, init='k-means++', random_state=42, n_init=1)
            kmeans.fit(data)
            centroids = kmeans.cluster_centers_
        else:
            centroids = data.copy()
    
    # Find centroid indices
    if len(centroids) > 0:
        try:
            centroid_distances = pairwise_distances(centroids, synthetic_candidates)
            centroid_indices = np.argmin(centroid_distances, axis=1)
        except Exception as e:
            print(f"Error finding centroid indices: {e}")
            centroid_indices = np.arange(len(centroids))
    else:
        centroid_indices = np.array([])
    
    # Visualization
    if synthetic_candidates.shape[1] == 2 and len(centroids) > 0:
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot energy landscape
            plt.scatter(synthetic_candidates[:, 0], synthetic_candidates[:, 1], 
                       c=candidate_energy, cmap='viridis', alpha=0.6, s=20)
            
            # Mark identified centroids
            plt.scatter(centroids[:, 0], centroids[:, 1],
                       marker='*', s=400, c='red', edgecolors='black',
                       label=f'K-means++ Centroids (n={len(centroids)})')
            
            plt.colorbar(label='Energy Level')
            plt.title("K-means++ Enhanced Centroid Detection")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    return centroids, centroid_indices




def SNN_optimized(nc: int, data: np.ndarray,candidates_multiplier, energy_multiplier,energy_threshold) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized SNN clustering using synthetic grid candidates"""
    n, d = data.shape
    distance = squareform(pdist(data))

    
    # Step 2: Energy calculation (on real data)
    eps = np.percentile(distance[distance > 0], 5)
    energy = np.sum(1/(distance**2 + eps**2), axis=1)
    
    # Step 3: Density-aware weighting
    log_energy = np.log(energy + 1e-10)
    weights = np.exp(log_energy - np.max(log_energy))
    
    # Step 4: Penalized centroid selection from synthetic candidates
    syn_centroids, centroid_indices = penalized_energy_centroids(nc,data,candidates_multiplier,energy_multiplier,energy_threshold)
    
    # For now, no assignments (set to None or zeros)
    indexAssignment = np.zeros(n, dtype=int)  # Placeholder
    
    # Visualization
    if d <= 3:
        plot_clusters(data, syn_centroids, energy, assignments=indexAssignment)
    
    return syn_centroids, centroid_indices

def plot_clusters(data, synthetic_centroids, energy, assignments=None):
    """Visualize the data points with synthetic centroids"""
    plt.figure(figsize=(12, 7))
    
    # Calculate energy threshold
    min_e, max_e = np.min(energy), np.max(energy)
    threshold = min_e
    high_energy_mask = energy > threshold
    
    # Plot all points (below threshold in blue)
    plt.scatter(data[~high_energy_mask, 0], data[~high_energy_mask, 1], 
                c='lightblue', alpha=0.4, s=30, label='Below energy threshold')
    
    # Plot high-energy points (red)
    plt.scatter(data[high_energy_mask, 0], data[high_energy_mask, 1], 
                c='red', alpha=0.6, s=30, label='Above energy threshold')
    
    # Plot cluster assignments if available and not all zeros
    if assignments is not None and not np.all(assignments == 0):
        # Override colors for assigned points
        scatter = plt.scatter(data[:, 0], data[:, 1], c=assignments, 
                             cmap='tab20', alpha=0.7, s=30, label='Cluster assignments')
        plt.colorbar(scatter, label='Cluster ID')
    
    # Plot synthetic centroids (large gold stars)
    plt.scatter(synthetic_centroids[:, 0], synthetic_centroids[:, 1],
                marker='*', s=400, c='gold', edgecolors='black',
                linewidths=2, label='Synthetic Centroids', zorder=5)
    
    # Add numbering to centroids
    for i, centroid in enumerate(synthetic_centroids):
        plt.annotate(f'{i+1}', (centroid[0], centroid[1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='black')
    
    plt.title(f'Synthetic Centroids Selection (Energy Threshold: {threshold:.2f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()








def add_laplace_noise_vectorized(data, epsilon, sensitivity):
    """Vectorized Laplace noise addition"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise


def nnfc_optimized(data_path, use_gpu=False,k1=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None,energy_threshold=None):
    """Optimized NNFC function"""

    datapkl = load_dataset(data_path)
    # eachlable = datapkl['eachlable']
    # order = datapkl['order']
    true_labels = np.array(datapkl['true_label'])  # Original labels for full data
    data = np.array(datapkl['full_data'])
    print(data.shape)
    
    corepoints = []
    # print('order',len(order))
    print('stage1 starting')
    
    for i_client in range(10):
        print('stage1 with the client',i_client)
        lodata = datapkl["client_" + str(i_client)]

        # reducer = umap.UMAP(n_components=2, random_state=42)
        # lodata = reducer.fit_transform(lodata)


        # noise = np.random.uniform(0, 1, size=lodata.shape)

        # euc = euclidean_distances(lodata)
        # row, col = np.diag_indices_from(euc)
        # euc[row, col] = np.max(euc)
        # minvalue = np.min(euc)
        # ratio = []
        # noisenew = noise
        # for i in range(noisenew.shape[0]):
        #     for j in range(noisenew.shape[1]):
        #         if noisenew[i][j] >= 0.5:
        #             noisenew[i][j] = 1 - noisenew[i][j]

        # arr = noisenew.reshape(1, noise.shape[0] * noise.shape[1])[0]
        # ratio = []
        # for i in arr:
        #     for j in arr:
        #         num1 = float(i)/float(j)
        #         num2 = float(j)/float(i)
        #         ratio.append(max([num1, num2]))    
        # maxratio = max(ratio)


        # epsilon = 1.01 * maxratio / minvalue
        # print(epsilon)
        
        # scale = 1 / epsilon
        # laplace_noise = np.random.laplace(loc=0.0, scale=scale, size=lodata.shape)
        # lodata_noisy = lodata + laplace_noise

        scale = 1 / epsilon
        laplace_noise = np.random.laplace(loc=0.0, scale=scale, size=lodata.shape)


        lodata_noisy = lodata + laplace_noise

    
    
        # n_clusters = min(len(lodata_noisy) // 3, 50)
        n_clusters = k1
        
      
        cluster = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        centers = cluster.fit(lodata_noisy).cluster_centers_
        
        corepoints.append(centers)
    
    print('stage2 starting')
    
    serverdata = np.concatenate(corepoints, axis=0)
    label = datapkl['true_label']
    # cnum = len(set(label))
    cnum = len(set(true_labels))
    # k = min(5, len(serverdata) // 10)  # Adaptive k
    # print('cnum',cnum)


    print('stage6 starting')


    # Old: finalcenter = serverdata[centroid]
    finalcenter, assignment = SNN_optimized(cnum, serverdata,candidates_multiplier,energy_multiplier,energy_threshold)  # Now returns synthetic centroids

    
    # Compute distances from all data points to synthetic centroids
    distances = cdist(data, finalcenter)  # Faster than manual loops

    # Assign each point to the nearest synthetic centroid
    idx = np.argmin(distances, axis=1) + 1  # +1 for 1-based indexing

    # Compute metrics
    ari = round(adjusted_rand_score(label, idx), 4)
    nmi = round(normalized_mutual_info_score(label, idx), 4)

    return ari, nmi, [n_clusters, cnum]



def run_single_experiment(args):
    """Single experiment runner for multiprocessing"""
    data_path, use_gpu, k1,candidates_multiplier,energy_multiplier,epsilon,energy_threshold = args
    return nnfc_optimized(data_path, use_gpu,k1,candidates_multiplier,energy_multiplier,epsilon,energy_threshold)






def run_experiments_parallel(data_path, n_runs=1000, n_processes=None, use_gpu=True, k1=None, dataset=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None,energy_threshold=None):
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

    batches = (n_runs + batch_size - 1) // batch_size  # Ceiling division

    run_counter = 0
    for batch_idx in range(batches):
        current_batch_size = min(batch_size, n_runs - run_counter)
        args_list = [(data_path, use_gpu, k1, candidates_multiplier,energy_multiplier,epsilon,energy_threshold) for i in range(current_batch_size)]

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            future_to_run = {executor.submit(run_single_experiment, args): run_counter + i for i, args in enumerate(args_list)}

            for future in as_completed(future_to_run):
                run_id = future_to_run[future]
                try:
                    result = future.result()
                    results.append(result)

                    if len(results) % 100 == 0 or dataset in ['celltypes','covtype','postures','mnist','bot']:
                        elapsed = time.time() - start_time
                        print(f"Completed {len(results)}/{n_runs} runs in {elapsed:.1f}s")
                        print('results_max',max(results))
                except Exception as e:
                    print(f"Run {run_id} failed: {e}")
                    results.append((0, 0, [0, 0, 0]))

        run_counter += current_batch_size

    return results







def evaluate_with_seeds(data_path, use_gpu=True, n_runs=100, n_processes=None,k1=None,dataset=None,seed=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None,energy_threshold=None):
    """Evaluate performance across 10 seeds"""
    SEEDS = list(range(seed))
    seed_results = {'ari_max': [], 'nmi_max': []}
    
    for seed in SEEDS:
        print(f"\n--- Evaluating with seed={seed} ---")
        results = run_experiments_parallel(data_path, n_runs, n_processes, use_gpu,k1,dataset,candidates_multiplier,energy_multiplier,epsilon,energy_threshold)
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
        candidates_multiplier = config[dataset]['candidates_multiplier']
        energy_multiplier = config[dataset]['energy_multiplier']
        epsilon = config['epsilon']
        energy_threshold = config[dataset]['energy_threshold']
     

        

        # Evaluate with 10 seeds
        seed_results = evaluate_with_seeds(data_path, use_gpu, n_runs=n_runs, n_processes=n_processes,k1=k1,dataset=dataset,seed=seeds,energy_multiplier=energy_multiplier,candidates_multiplier=candidates_multiplier,epsilon=epsilon,energy_threshold=energy_threshold)
        
        # Save seed results
        with open(save_path, 'a') as f:
            f.write(f"{data_path}\n")
            f.write(f"ARI (max) across seeds: {seed_results['ari_max']}\n")
            f.write(f"NMI (max) across seeds: {seed_results['nmi_max']}\n")
            f.write(f"Mean ± Std - ARI: {np.mean(seed_results['ari_max']):.4f} ± {np.std(seed_results['ari_max']):.4f}\n")
            f.write(f"Mean ± Std - NMI: {np.mean(seed_results['nmi_max']):.4f} ± {np.std(seed_results['nmi_max']):.4f}\n\n")






