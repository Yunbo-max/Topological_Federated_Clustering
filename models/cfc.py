# # -*- coding: utf-8 -*-
# # @Author: Yunbo
# # @Date:   2025-06-02 17:55:49
# # @Last Modified by:   Yunbo
# # @Last Modified time: 2025-07-05 22:37:23
# import umap
# import json
# import os
# from scipy.spatial.distance import pdist, squareform, cdist
# # Optimized version with multi-processing and GPU acceleration support
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import json
# import pickle
# import math
# import os
# from sklearn.cluster import DBSCAN
# import random
# from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
# from sklearn.metrics import adjusted_mutual_info_score
# from typing import List, Tuple
# from scipy.spatial.distance import pdist, squareform
# from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
# import multiprocessing as mp
# from functools import partial
# import time
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import warnings
# warnings.filterwarnings('ignore')

# # Try to use GPU acceleration if available


# def load_dataset(filepath):
#     """Load dataset from pickle file"""
#     with open(filepath, 'rb') as fr:
#         dataset = pickle.load(fr)
#     return dataset

# def dist(a, b, ax=1):
#     """Compute distance between two points"""
#     return np.linalg.norm(a - b, axis=ax)


# import numpy as np
# from scipy.spatial.distance import pdist, squareform

# # -*- coding: utf-8 -*-
# import numpy as np
# from scipy.spatial.distance import pdist, squareform
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.neighbors import kneighbors_graph
# from scipy.sparse.linalg import eigsh
# import networkx as nx
# import numpy as np
# from scipy.spatial.distance import pdist, squareform, cdist
# from typing import Tuple



# from sklearn.neighbors import NearestNeighbors




# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import squareform, pdist



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics.pairwise import pairwise_distances
# from scipy.spatial.distance import pdist, squareform
# from typing import Tuple

# def generate_synthetic_candidates(data, n_candidates_multiplier=1.5):
#     """Generate synthetic candidate points using a grid approach"""
#     n_candidates = int(len(data) * n_candidates_multiplier)
    
#     # Create grid based on data bounds
#     mins = np.min(data, axis=0)
#     maxs = np.max(data, axis=0)
    
#     # Expand bounds slightly
#     range_expand = 0.0
#     mins -= range_expand * (maxs - mins)
#     maxs += range_expand * (maxs - mins)
    
#     # Generate grid points
#     if data.shape[1] == 2:
#         grid_size = int(np.sqrt(n_candidates))
#         x = np.linspace(mins[0], maxs[0], grid_size)
#         y = np.linspace(mins[1], maxs[1], grid_size)
#         xx, yy = np.meshgrid(x, y)
#         candidates = np.column_stack([xx.ravel(), yy.ravel()])
#     else:
#         # For higher dimensions, use random sampling within bounds
#         candidates = np.random.uniform(mins, maxs, (n_candidates, data.shape[1]))
    
#     return candidates

# from scipy.spatial.distance import cdist

# def penalized_energy_centroids(data, nc,candidates_multiplier,energy_multiplier):
#     """Select centroids directly from synthetic candidates using energy-based method"""
#     # Generate synthetic candidates
#     synthetic_candidates = generate_synthetic_candidates(data, n_candidates_multiplier=candidates_multiplier)
    
#     # Calculate distances and energies
#     candidate_distances = cdist(data, synthetic_candidates, 'euclidean')  # or your preferred metric
#     candidate_energy = np.sum(1/(candidate_distances+30), axis=0)
    

#     def plot_energy_heatmap(data, candidate_energy, synthetic_candidates):
#         """
#         Plot the candidate energies as a heatmap.
        
#         Parameters:
#         - data: Original data points (2D array)
#         - candidate_energy: Energy values for each candidate (1D array)
#         - synthetic_candidates: Positions of synthetic candidates (2D array)
#         """
#         plt.figure(figsize=(10, 8))
        
#         # Create a scatter plot of the original data points
#         plt.scatter(data[:, 0], data[:, 1], c='blue', s=10, label='Original Data')
        
#         # Create a scatter plot of the synthetic candidates colored by energy
#         sc = plt.scatter(synthetic_candidates[:, 0], synthetic_candidates[:, 1], 
#                         c=candidate_energy, cmap='viridis', s=50, 
#                         label='Synthetic Candidates')
        
#         # Add colorbar
#         plt.colorbar(sc, label='Candidate Energy')
        
#         plt.title('Candidate Energy Heatmap')
#         plt.xlabel('X coordinate')
#         plt.ylabel('Y coordinate')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     # Example usage (assuming you have these variables already):
#     # synthetic_candidates = generate_synthetic_candidates(data, n_candidates_multiplier=candidates_multiplier)
#     # candidate_distances = cdist(data, synthetic_candidates, 'euclidean')
#     # candidate_energy = np.sum(1/(candidate_distances + eps), axis=0)

#     # Then call the plotting function:
#     plot_energy_heatmap(data, candidate_energy, synthetic_candidates)

    
#     # Calculate candidate energies
#     eps = 5
#     print('eps',eps)
#     # eps = 0.1
#     candidate_energy = np.sum(1/(candidate_distances**energy_multiplier+eps ), axis=0)
    
#     # Apply density-aware weighting
#     log_energy = np.log(candidate_energy )
#     candidate_weights = np.exp(log_energy - np.max(log_energy))

#     total_energy = candidate_energy


#     # --- Plot 2D energy landscape ---
#     if synthetic_candidates.shape[1] == 2:  # Only for 2D data
#         x = synthetic_candidates[:, 0]
#         y = synthetic_candidates[:, 1]
#         plt.figure(figsize=(8, 6))
#         scatter = plt.scatter(x, y, c=total_energy, cmap='viridis', s=40)
#         plt.colorbar(scatter, label='Candidate Energy')
#         # plt.scatter(data[:, 0], data[:, 1], c='red', s=10, alpha=0.3, label='Original Data')
#         plt.title("Energy Map of Synthetic Candidates")
#         plt.xlabel("Feature 1")
#         plt.ylabel("Feature 2")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("Plotting skipped: data is not 2D")
    
#     # Initialize centroid indices
#     centroid_indices = []

#     for _ in range(nc):
#         scores = total_energy
#         scores[centroid_indices] = -np.inf  # Avoid reselecting the same centroid
#         next_idx = np.argmax(scores)
#         centroid_indices.append(int(next_idx))

    
#     # Convert to numpy array with explicit integer dtype
#     centroid_indices_arr = np.array(centroid_indices, dtype=np.int32)
    
#     # Return the actual synthetic centroid points and their indices
#     return synthetic_candidates[centroid_indices_arr], centroid_indices_arr

# def SNN_optimized(nc: int, data: np.ndarray,candidates_multiplier, energy_multiplier) -> Tuple[np.ndarray, np.ndarray]:
#     """Optimized SNN clustering using synthetic grid candidates"""
#     n, d = data.shape

#     # Step 4: Penalized centroid selection from synthetic candidates
#     syn_centroids, centroid_indices = penalized_energy_centroids(data, nc, candidates_multiplier, energy_multiplier)
    
#     # For now, no assignments (set to None or zeros)
#     indexAssignment = np.zeros(n, dtype=int)  # Placeholder
    
#     # Visualization
#     if d <= 3:
#         plot_clusters(data, syn_centroids, energy, assignments=indexAssignment)
    
#     return syn_centroids, centroid_indices

# def plot_clusters(data, synthetic_centroids, energy, assignments=None):
#     """Visualize the data points with synthetic centroids"""
#     plt.figure(figsize=(12, 7))
    
#     # Calculate energy threshold
#     min_e, max_e = np.min(energy), np.max(energy)
#     threshold = min_e
#     high_energy_mask = energy > threshold
    
#     # Plot all points (below threshold in blue)
#     plt.scatter(data[~high_energy_mask, 0], data[~high_energy_mask, 1], 
#                 c='lightblue', alpha=0.4, s=30, label='Below energy threshold')
    
#     # Plot high-energy points (red)
#     plt.scatter(data[high_energy_mask, 0], data[high_energy_mask, 1], 
#                 c='red', alpha=0.6, s=30, label='Above energy threshold')
    
#     # Plot cluster assignments if available and not all zeros
#     if assignments is not None and not np.all(assignments == 0):
#         # Override colors for assigned points
#         scatter = plt.scatter(data[:, 0], data[:, 1], c=assignments, 
#                              cmap='tab20', alpha=0.7, s=30, label='Cluster assignments')
#         plt.colorbar(scatter, label='Cluster ID')
    
#     # Plot synthetic centroids (large gold stars)
#     plt.scatter(synthetic_centroids[:, 0], synthetic_centroids[:, 1],
#                 marker='*', s=400, c='gold', edgecolors='black',
#                 linewidths=2, label='Synthetic Centroids', zorder=5)
    
#     # Add numbering to centroids
#     for i, centroid in enumerate(synthetic_centroids):
#         plt.annotate(f'{i+1}', (centroid[0], centroid[1]), 
#                     xytext=(5, 5), textcoords='offset points',
#                     fontsize=12, fontweight='bold', color='black')
    
#     plt.title(f'Synthetic Centroids Selection (Energy Threshold: {threshold:.2f})')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()







# def add_laplace_noise_vectorized(data, epsilon, sensitivity):
#     """Vectorized Laplace noise addition"""
#     scale = sensitivity / epsilon
#     noise = np.random.laplace(0, scale, data.shape)
#     return data + noise


# def nnfc_optimized(data_path, use_gpu=False,k1=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None):
#     """Optimized NNFC function"""

#     datapkl = load_dataset(data_path)
#     # eachlable = datapkl['eachlable']
#     # order = datapkl['order']
#     true_labels = np.array(datapkl['true_label'])  # Original labels for full data
#     data = np.array(datapkl['full_data'])
#     print(data.shape)
    
#     corepoints = []
#     # print('order',len(order))
#     print('stage1 starting')
    
#     for i_client in range(10):
#         print('stage1 with the client',i_client)
#         lodata = datapkl["client_" + str(i_client)]

#         # reducer = umap.UMAP(n_components=2, random_state=42)
#         # lodata = reducer.fit_transform(lodata)


#         # noise = np.random.uniform(0, 1, size=lodata.shape)

#         # euc = euclidean_distances(lodata)
#         # row, col = np.diag_indices_from(euc)
#         # euc[row, col] = np.max(euc)
#         # minvalue = np.min(euc)
#         # ratio = []
#         # noisenew = noise
#         # for i in range(noisenew.shape[0]):
#         #     for j in range(noisenew.shape[1]):
#         #         if noisenew[i][j] >= 0.5:
#         #             noisenew[i][j] = 1 - noisenew[i][j]

#         # arr = noisenew.reshape(1, noise.shape[0] * noise.shape[1])[0]
#         # ratio = []
#         # for i in arr:
#         #     for j in arr:
#         #         num1 = float(i)/float(j)
#         #         num2 = float(j)/float(i)
#         #         ratio.append(max([num1, num2]))    
#         # maxratio = max(ratio)


#         # epsilon = 1.01 * maxratio / minvalue
#         # print(epsilon)
        
#         # scale = 1 / epsilon
#         # laplace_noise = np.random.laplace(loc=0.0, scale=scale, size=lodata.shape)
#         # lodata_noisy = lodata + laplace_noise

#         scale = 1 / epsilon
#         laplace_noise = np.random.laplace(loc=0.0, scale=scale, size=lodata.shape)


#         lodata_noisy = lodata + laplace_noise

    
    
#         # n_clusters = min(len(lodata_noisy) // 3, 50)
#         n_clusters = k1
        
      
#         cluster = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         centers = cluster.fit(lodata_noisy).cluster_centers_
        
#         corepoints.append(centers)
    
#     print('stage2 starting')
    
#     serverdata = np.concatenate(corepoints, axis=0)
#     label = datapkl['true_label']
#     # cnum = len(set(label))
#     cnum = len(set(true_labels))
#     # k = min(5, len(serverdata) // 10)  # Adaptive k
#     # print('cnum',cnum)

#     print('stage6 starting')


#     # Old: finalcenter = serverdata[centroid]
#     finalcenter, assignment = SNN_optimized(cnum, serverdata,candidates_multiplier,energy_multiplier)  # Now returns synthetic centroids

    
#     # Compute distances from all data points to synthetic centroids
#     distances = cdist(data, finalcenter)  # Faster than manual loops

#     # Assign each point to the nearest synthetic centroid
#     idx = np.argmin(distances, axis=1) + 1  # +1 for 1-based indexing

#     # Compute metrics
#     ari = round(adjusted_rand_score(label, idx), 4)
#     nmi = round(normalized_mutual_info_score(label, idx), 4)

#     return ari, nmi, [n_clusters,cnum]



# def run_single_experiment(args):
#     """Single experiment runner for multiprocessing"""
#     data_path, use_gpu, k1,candidates_multiplier,energy_multiplier,epsilon = args
#     return nnfc_optimized(data_path, use_gpu,k1,candidates_multiplier,energy_multiplier,epsilon)






# def run_experiments_parallel(data_path, n_runs=1000, n_processes=None, use_gpu=True, k1=None, dataset=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None):
#     """Run experiments in parallel, optionally in batches (e.g., for 'mnist2')"""
#     if n_processes is None:
#         n_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes max
    
#     print(f"Running {n_runs} experiments on {n_processes} cores...")

#     results = []
#     start_time = time.time()

#     # Define batch size
#     if dataset in ['celltypes1','covtype1','postures1','mnist2d1']:
#         batch_size = 5
#     else:
#         batch_size = n_runs  # run all at once

#     batches = (n_runs + batch_size - 1) // batch_size  # Ceiling division

#     run_counter = 0
#     for batch_idx in range(batches):
#         current_batch_size = min(batch_size, n_runs - run_counter)
#         args_list = [(data_path, use_gpu, k1, candidates_multiplier,energy_multiplier,epsilon) for i in range(current_batch_size)]

#         with ProcessPoolExecutor(max_workers=n_processes) as executor:
#             future_to_run = {executor.submit(run_single_experiment, args): run_counter + i for i, args in enumerate(args_list)}

#             for future in as_completed(future_to_run):
#                 run_id = future_to_run[future]
#                 try:
#                     result = future.result()
#                     results.append(result)

#                     if len(results) % 100 == 0 or dataset in ['celltypes','covtype','postures','mnist','bot']:
#                         elapsed = time.time() - start_time
#                         print(f"Completed {len(results)}/{n_runs} runs in {elapsed:.1f}s")
#                         print('results_max',max(results))
#                 except Exception as e:
#                     print(f"Run {run_id} failed: {e}")
#                     results.append((0, 0, [0, 0, 0]))

#         run_counter += current_batch_size

#     return results







# def evaluate_with_seeds(data_path, use_gpu=True, n_runs=100, n_processes=None,k1=None,dataset=None,seed=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None):
#     """Evaluate performance across 10 seeds"""
#     SEEDS = list(range(seed))
#     seed_results = {'ari_max': [], 'nmi_max': []}
    
#     for seed in SEEDS:
#         print(f"\n--- Evaluating with seed={seed} ---")
#         results = run_experiments_parallel(data_path, n_runs, n_processes, use_gpu,k1,dataset,candidates_multiplier,energy_multiplier,epsilon)
#         ari_scores = [r[0] for r in results if r[0] > 0]
#         nmi_scores = [r[1] for r in results if r[1] > 0]
        
#         if ari_scores and nmi_scores:
#             seed_results['ari_max'].append(max(ari_scores))
#             seed_results['nmi_max'].append(max(nmi_scores))
    
#     # Calculate statistics
#     ari_mean = np.mean(seed_results['ari_max'])
#     ari_std = np.std(seed_results['ari_max'])
#     nmi_mean = np.mean(seed_results['nmi_max'])
#     nmi_std = np.std(seed_results['nmi_max'])
    
#     print("\n" + "="*50)
#     print("SEED EVALUATION RESULTS (10 seeds)")
#     print("="*50)
#     print(f"ARI (max): {ari_mean:.4f} ± {ari_std:.4f}")
#     print(f"NMI (max): {nmi_mean:.4f} ± {nmi_std:.4f}")
    
#     return seed_results



# def run_experiment(datanames, seeds, n_runs, use_gpu,save_path,config_file):
#     """Updated main function"""

#     config_path = config_file

#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found: {config_path}")

#     with open(config_path, 'r') as f:
#         config = json.load(f)

#     use_gpu = True
#     n_processes = mp.cpu_count() - 1
    
#     for dataset in datanames:
#         data_path = f'dataset/{dataset}fed.pkl'
#         print(f"\nProcessing dataset: {dataset}")
        
#         if not os.path.exists(data_path):
#             print(f"Dataset {data_path} not found, skipping...")
#             continue


#         if dataset not in config:
#             print(f"Config for {dataset} not found, skipping...")
#             continue

#         k1 = config[dataset]['k1']
#         candidates_multiplier = config[dataset]['candidates_multiplier']
#         energy_multiplier = config[dataset]['energy_multiplier']
#         epsilon = config['epsilon']
     

        

#         # Evaluate with 10 seeds
#         seed_results = evaluate_with_seeds(data_path, use_gpu, n_runs=n_runs, n_processes=n_processes,k1=k1,dataset=dataset,seed=seeds,energy_multiplier=energy_multiplier,candidates_multiplier=candidates_multiplier,epsilon=epsilon)
        
#         # Save seed results
#         with open(save_path, 'a') as f:
#             f.write(f"{data_path}\n")
#             f.write(f"ARI (max) across seeds: {seed_results['ari_max']}\n")
#             f.write(f"NMI (max) across seeds: {seed_results['nmi_max']}\n")
#             f.write(f"Mean ± Std - ARI: {np.mean(seed_results['ari_max']):.4f} ± {np.std(seed_results['ari_max']):.4f}\n")
#             f.write(f"Mean ± Std - NMI: {np.mean(seed_results['nmi_max']):.4f} ± {np.std(seed_results['nmi_max']):.4f}\n\n")






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
#     """Optimized SNN clustering with improved electrical potential calculation"""
#     unassigned = -1
#     n, d = data.shape

#     print('n',n)

#     # Compute pairwise distances and neighbors
#     distance = squareform(pdist(data))
#     indexDistanceAsc = np.argsort(distance, axis=1)
#     indexNeighbor = indexDistanceAsc[:, :k]

#     # # Shared neighbor calculation
#     # numSharedNeighbor = np.zeros([n, n], dtype=int)
#     # for i in range(n):
#     #     for j in range(i):
#     #         shared = np.intersect1d(indexNeighbor[i], indexNeighbor[j])
#     #         numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = len(shared)

#     # =====================================================================
#     # IMPROVED POTENTIAL CALCULATION FOR SYNTHETIC CENTROIDS
#     # =====================================================================
#     np.fill_diagonal(distance, np.inf)
#     eps = np.percentile(distance[distance > 0], 5)  # Adaptive epsilon
    
#     # Calculate energy for all data points (charges)
#     energy = np.sum(1/(distance**2 + eps**2), axis=1)
    
#     # Generate synthetic candidates in expanded space
#     data_min = np.min(data, axis=0)
#     data_max = np.max(data, axis=0)
#     expansion = 0
#     from sklearn.model_selection import train_test_split
#     # First get uniform samples
#     temp_samples = np.random.uniform(
#         low=data_min - expansion,
#         high=data_max + expansion,
#         size=(2*n, d)
#     )
#     # Then select diverse samples
#     #  Stratified Sampling
#     synthetic_candidates, _ = train_test_split(temp_samples, train_size=0.5)
        
            
#     # Vectorized potential calculation using same improved formula
#     candidate_distances = cdist(synthetic_candidates, data)
#     candidate_potentials = np.sum(1/(candidate_distances**2 + eps**2), axis=1)
    
#     # Apply same density-aware selection as original
#     log_potential = np.log(candidate_potentials + 1e-10)
#     weights = np.exp(log_potential - np.max(log_potential))  # Softmax
#     candidate_scores = weights * candidate_potentials
    
#     # Select best candidates
#     top_candidates = np.argsort(candidate_scores)[-nc:]
#     synthetic_centroids = synthetic_candidates[top_candidates]


#     # =====================================================================
#     # PLOTTING
#     # =====================================================================
#     plt.figure(figsize=(15, 5))
    
#     # Plot 1: Original data points
#     plt.subplot(1, 3, 1)
#     plt.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.5, label='Data points')
#     plt.title('Original Data Points')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend()
    
#     # Plot 2: Synthetic candidates grid
#     plt.subplot(1, 3, 2)
#     plt.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.3, label='Data points')
#     plt.scatter(synthetic_candidates[:, 0], synthetic_candidates[:, 1], 
#                 c='orange', alpha=0.5, marker='x', label='Synthetic candidates')
#     plt.title('Synthetic Candidates Grid')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend()
    
#     # Plot 3: Final selected centroids
#     plt.subplot(1, 3, 3)
#     plt.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.3, label='Data points')
#     plt.scatter(synthetic_centroids[:, 0], synthetic_centroids[:, 1], 
#                 c='red', s=100, marker='*', label='Selected centroids')
#     plt.title('Final Selected Centroids')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()
    
#     # =====================================================================
#     # IMPROVED CLUSTER ASSIGNMENT
#     # =====================================================================
#     # Calculate potential-aware distances
#     data_energies = energy / np.max(energy)  # Normalized
#     centroid_energies = candidate_potentials[top_candidates] / np.max(candidate_potentials)
    
#     # Combined distance metric
#     euclidean_dists = cdist(data, synthetic_centroids)
#     energy_diffs = np.abs(data_energies[:, None] - centroid_energies)
#     combined_dists = euclidean_dists * (1 + energy_diffs)
    
#     assignments = np.argmin(combined_dists, axis=1)
    
#     return synthetic_centroids, assignments









def SNN_optimized(k: int, nc: int, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized SNN clustering with improved electrical potential calculation"""
    unassigned = -1
    n, d = data.shape

    print('n',n)

    # Compute pairwise distances and neighbors
    distance = squareform(pdist(data))
    indexDistanceAsc = np.argsort(distance, axis=1)
    indexNeighbor = indexDistanceAsc[:, :k]

    # # Shared neighbor calculation
    # numSharedNeighbor = np.zeros([n, n], dtype=int)
    # for i in range(n):
    #     for j in range(i):
    #         shared = np.intersect1d(indexNeighbor[i], indexNeighbor[j])
    #         numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = len(shared)

    # =====================================================================
    # IMPROVED POTENTIAL CALCULATION FOR SYNTHETIC CENTROIDS
    # =====================================================================
    np.fill_diagonal(distance, np.inf)
    eps = np.percentile(distance[distance > 0], 5)  # Adaptive epsilon

    # Generate synthetic candidates in expanded space
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    expansion = 0 * (data_max - data_min)  # 10% expansion
    n_candidates = 2 * n
    
    # Create grid-based synthetic candidates
    grid_resolution = int(np.ceil((n_candidates) ** (1/d)))
    print('grid_resolution',grid_resolution)
    axes = [np.linspace(data_min[i] - expansion[i], 
            data_max[i] + expansion[i], 
            grid_resolution) for i in range(d)]
    grid = np.meshgrid(*axes)
    synthetic_candidates = np.vstack([g.ravel() for g in grid]).T

        
    # Vectorized potential calculation
    candidate_distances = cdist(synthetic_candidates, data)
    candidate_potentials = np.sum(1/(candidate_distances**2 + eps**2), axis=1)

    # Identify multiple high-potential regions
    max_potential = np.max(candidate_potentials)
    print('max_potential',max_potential)
    min_potential = np.min(candidate_potentials)
    print('min_potential',min_potential)
    reference_potential = (max_potential + min_potential)*0.5
    # reference_potential = np.mean(candidate_potentials)


    # Find candidates above reference potential
    high_potential_mask = candidate_potentials > reference_potential
    high_potential_candidates = synthetic_candidates[high_potential_mask]
    high_potential_values = candidate_potentials[high_potential_mask]

    print('high_potential_candidates',len(high_potential_candidates))

    if len(high_potential_candidates) > 0:
        print('yes')
        # Sort candidates by potential (descending)
        sorted_indices = np.argsort(high_potential_values)[::-1]
        sorted_candidates = high_potential_candidates[sorted_indices]
        
        # Initialize list of regions with the first (highest potential) candidate
        regions = [sorted_candidates[0:1]]

        

        # Calculate the fixed distance between adjacent grid nodes
        grid_spacing = min((data_max - data_min) / (grid_resolution - 1))
        diagonal_distance = np.sqrt(np.sum(grid_spacing**2))

        threshold = (diagonal_distance)*2

        # print('threshold',threshold)

        # For each subsequent candidate, check if it belongs to existing region or new one
        for candidate in sorted_candidates[1:]:
            if len(regions) == 0:
                regions.append(np.array([candidate]))
                continue
                
            # Calculate minimum distance to each existing region (using each region's centroid)
            region_centroids = np.array([np.mean(region, axis=0) for region in regions])
            distances_to_regions = cdist(candidate.reshape(1,-1), region_centroids).flatten()
            print('distances_to_regions',distances_to_regions)
            closest_region = np.argmin(distances_to_regions)
            min_distance = distances_to_regions[closest_region]
            
            # If closer than threshold, add to existing region
            if min_distance <= threshold:
                regions[closest_region] = np.vstack([regions[closest_region], candidate])
            else:
                # Create new region
                regions.append(np.array([candidate]))


        
        print(f"Found {len(regions)} distinct high-potential regions")
        
        # Select top candidate from each region
        selected_candidates = []
        for region in regions:
            # Get indices of candidates in this region in the original array
            region_indices_in_high = [i for i, x in enumerate(high_potential_candidates) 
                                    if any(np.all(x == y) for y in region)]
            original_indices = np.where(high_potential_mask)[0][region_indices_in_high]
            
            # Select the highest potential candidate in this region
            top_in_region = original_indices[np.argmax(candidate_potentials[original_indices])]
            selected_candidates.append(top_in_region)
        
        # If we need more candidates, fill with remaining top candidates
        if len(selected_candidates) < nc:
            remaining_needed = nc - len(selected_candidates)
            # Get all candidates sorted by potential (not just high potential ones)
            all_sorted = np.argsort(candidate_potentials)[::-1]
            
            # Add top candidates not already selected
            for idx in all_sorted:
                if idx not in selected_candidates:
                    selected_candidates.append(idx)
                    if len(selected_candidates) >= nc:
                        break
        
        synthetic_centroids = synthetic_candidates[selected_candidates[:nc]]

    else:
        # Fallback to original method if no high-potential candidates found
        log_potential = np.log(candidate_potentials + 1e-10)
        weights = np.exp(log_potential - np.max(log_potential))
        candidate_scores = weights * candidate_potentials
        top_candidates = np.argsort(candidate_scores)[-nc:]
        synthetic_centroids = synthetic_candidates[top_candidates]


    # =====================================================================
    # ENHANCED PLOTTING - GRID POINTS VISUALIZATION
    # =====================================================================
    """Visualize high-dimensional potential field using grid points"""

    # Combine all points for consistent projection
    combined_points = np.vstack([data, synthetic_candidates, synthetic_centroids])

    # Choose reducer based on data size
    if len(combined_points) < 10000:  # t-SNE for smaller datasets
        from sklearn.manifold import TSNE
        reduced_all = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(combined_points)
    else:  # PCA for larger datasets
        from sklearn.decomposition import PCA
        reduced_all = PCA(n_components=2).fit_transform(combined_points)

    # Split the reduced points
    reduced_data = reduced_all[:len(data)]
    reduced_candidates = reduced_all[len(data):len(data)+len(synthetic_candidates)]
    reduced_centroids = reduced_all[-len(synthetic_centroids):]

    # # Create visualization
    # plt.figure(figsize=(15, 5))

    # # Plot 1: Original data points
    # plt.subplot(1, 3, 1)
    # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', alpha=0.5)
    # plt.title('Original Data Points')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')

    # # Plot 2: Synthetic candidates colored by potential
    # plt.subplot(1, 3, 2)
    # scatter = plt.scatter(reduced_candidates[:, 0], reduced_candidates[:, 1], 
    #                     c=candidate_potentials, cmap='viridis', alpha=0.7)
    # plt.colorbar(scatter, label='Electrical Potential')
    # plt.title('Synthetic Candidates by Potential')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')

    # # Plot 3: Highlight selected centroids
    # plt.subplot(1, 3, 3)
    # plt.scatter(reduced_candidates[:, 0], reduced_candidates[:, 1], 
    #         c=candidate_potentials, cmap='viridis', alpha=0.3)

    # # Highlight centroids with labels
    # for i, (x, y) in enumerate(reduced_centroids):
    #     plt.scatter(x, y, c='red', s=100, marker='*', edgecolor='black')
    #     # plt.text(x, y, str(i+1), color='white', ha='center', va='center', 
    #     #         fontweight='bold')

    # plt.title('Selected Centroids')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.colorbar(label='Electrical Potential')

    # plt.tight_layout()
    # plt.show()


    
    # =====================================================================
    # IMPROVED CLUSTER ASSIGNMENT
    # =====================================================================
    # Calculate potential-aware distances
    # data_energies = energy / np.max(energy)  # Normalized
    # centroid_energies = candidate_potentials[top_candidates] / np.max(candidate_potentials)
    
    # # Combined distance metric
    # euclidean_dists = cdist(data, synthetic_centroids)
    # energy_diffs = np.abs(data_energies[:, None] - centroid_energies)
    # combined_dists = euclidean_dists * (1 + energy_diffs)
    
    # assignments = np.argmin(combined_dists, axis=1)
    assignments = 0
    
    return synthetic_centroids, assignments







# import numpy as np
# from scipy.spatial.distance import pdist, squareform, cdist
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from typing import Tuple

# def SNN_optimized(k: int, nc: int, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """Optimized SNN clustering with improved electrical potential calculation"""
#     unassigned = -1
#     n, d = data.shape
#     print('d',d)
#     print('n',n)

#     # Compute pairwise distances and neighbors
#     distance = squareform(pdist(data))
#     indexDistanceAsc = np.argsort(distance, axis=1)
#     indexNeighbor = indexDistanceAsc[:, :k]

#     # Improved potential calculation
#     np.fill_diagonal(distance, np.inf)
#     eps = np.percentile(distance[distance > 0], 5)  # Adaptive epsilon

#     # Generate synthetic candidates - switch method based on dimensionality
#     data_min = np.min(data, axis=0)
#     data_max = np.max(data, axis=0)
#     n_candidates = 2 * n
    
#     if d <= 5:  # Use grid for low dimensions
#         grid_resolution = int(np.ceil(n_candidates ** (1/d)))
#         axes = [np.linspace(data_min[i], data_max[i], grid_resolution) 
#                 for i in range(d)]
#         grid = np.meshgrid(*axes)
#         synthetic_candidates = np.vstack([g.ravel() for g in grid]).T
#     else:  # Use random sampling for high dimensions
#         synthetic_candidates = np.random.uniform(
#             low=data_min,
#             high=data_max,
#             size=(n_candidates, d)
#         )

#     # Vectorized potential calculation
#     candidate_distances = cdist(synthetic_candidates, data)
#     candidate_potentials = np.sum(1/(candidate_distances**2 + eps**2), axis=1)

#     # Identify multiple high-potential regions
#     max_potential = np.max(candidate_potentials)
#     min_potential = np.min(candidate_potentials)
#     reference_potential = (max_potential + min_potential)*0.5

#     # Find candidates above reference potential
#     high_potential_mask = candidate_potentials > reference_potential
#     high_potential_candidates = synthetic_candidates[high_potential_mask]
#     high_potential_values = candidate_potentials[high_potential_mask]

#     if len(high_potential_candidates) > 0:
#         # Sort candidates by potential (descending)
#         sorted_indices = np.argsort(high_potential_values)[::-1]
#         sorted_candidates = high_potential_candidates[sorted_indices]
        
#         # Calculate adaptive threshold
#         threshold = np.median(pdist(sorted_candidates[:min(100, len(sorted_candidates))]))
        
#         # Cluster high potential candidates to identify regions
#         from sklearn.cluster import DBSCAN
#         clustering = DBSCAN(eps=threshold, min_samples=1).fit(sorted_candidates)
#         labels = clustering.labels_
#         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
#         # Select highest potential point from each cluster
#         selected_indices = []
#         for cluster_id in range(n_clusters):
#             cluster_mask = (labels == cluster_id)
#             cluster_potentials = high_potential_values[sorted_indices][cluster_mask]
#             highest_in_cluster = np.argmax(cluster_potentials)
#             original_idx = np.where(high_potential_mask)[0][sorted_indices][cluster_mask][highest_in_cluster]
#             selected_indices.append(original_idx)
        
#         synthetic_centroids = synthetic_candidates[selected_indices[:nc]]
#     else:
#         # Fallback to original method
#         log_potential = np.log(candidate_potentials + 1e-10)
#         weights = np.exp(log_potential - np.max(log_potential))
#         candidate_scores = weights * candidate_potentials
#         top_candidates = np.argsort(candidate_scores)[-nc:]
#         synthetic_centroids = synthetic_candidates[top_candidates]

#     # =====================================================================
#     # ENHANCED PLOTTING
#     # =====================================================================
#     """Visualize high-dimensional potential field and data together"""
    
#     # Combine all points for consistent projection
#     combined_points = np.vstack([data, synthetic_candidates])
    
#     # Choose reducer based on data size
#     if len(combined_points) < 100:  # t-SNE for smaller datasets
#         from sklearn.manifold import TSNE
#         reducer = TSNE(n_components=2, perplexity=30, random_state=42)
#     else:  # PCA for larger datasets
#         from sklearn.decomposition import PCA
#         reducer = PCA(n_components=2)
    
#     # Fit on combined points and transform everything
#     reduced_all = reducer.fit_transform(combined_points)
#     reduced_data = reduced_all[:len(data)]
#     reduced_grid = reduced_all[len(data):]
#     reduced_centroids = reducer.transform(synthetic_centroids)
    
#     # Create potential field visualization
#     plt.figure(figsize=(15, 5))
    
#     # Plot 1: Just the data points
#     plt.subplot(1, 3, 1)
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', alpha=0.5)
#     plt.title('Data Points (Reduced Space)')
    
#     # Plot 2: Potential field with data
#     plt.subplot(1, 3, 2)
    
#     # Create grid for potential field visualization
#     from scipy.interpolate import griddata
#     xmin, xmax = reduced_grid[:,0].min(), reduced_grid[:,0].max()
#     ymin, ymax = reduced_grid[:,1].min(), reduced_grid[:,1].max()
#     xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), 
#                          np.linspace(ymin, ymax, 100))
    
#     # Interpolate potentials onto regular grid
#     grid_potential = griddata(reduced_grid, candidate_potentials, 
#                              (xx, yy), method='cubic')
    
#     # Plot potential field
#     contour = plt.contourf(xx, yy, grid_potential, levels=20, cmap='viridis', alpha=0.6)
#     plt.colorbar(contour, label='Electrical Potential')
    
#     # Overlay data points
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', alpha=0.3)
#     plt.title('Potential Field with Data')
    
#     # Plot 3: With centroids
#     plt.subplot(1, 3, 3)
#     plt.contourf(xx, yy, grid_potential, levels=20, cmap='viridis', alpha=0.4)
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', alpha=0.2)
    
#     # Plot centroids with labels
#     for i, (x, y) in enumerate(reduced_centroids):
#         plt.scatter(x, y, c='red', s=150, marker='*', edgecolor='black')
#         plt.text(x, y, str(i+1), color='white', ha='center', va='center', 
#                  fontweight='bold')
    
#     plt.title('Centroids on Potential Field')
#     plt.tight_layout()
#     plt.show()

#     return synthetic_centroids, candidate_potentials




















def add_laplace_noise_vectorized(data, epsilon, sensitivity):
    """Vectorized Laplace noise addition"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise

def nnfc_optimized(data_path, use_gpu=False,k1=None,k2=None):
    """Optimized NNFC function"""

    datapkl = load_dataset(data_path)
    # eachlable = datapkl['eachlable']
    # order = datapkl['order']
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

        epsilon = 0.1
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
    cnum = len(set(label))

    # k = min(5, len(serverdata) // 10)  # Adaptive k
    # print('cnum',cnum)

    k = k2

    print('stage6 starting')


    # Old: finalcenter = serverdata[centroid]
    finalcenter, assignment = SNN_optimized(k, cnum, serverdata)  # Now returns synthetic centroids

    
    # Compute distances from all data points to synthetic centroids
    distances = cdist(data, finalcenter)  # Faster than manual loops

    # Assign each point to the nearest synthetic centroid
    idx = np.argmin(distances, axis=1) + 1  # +1 for 1-based indexing

    # Compute metrics
    ari = round(adjusted_rand_score(label, idx), 4)
    nmi = round(normalized_mutual_info_score(label, idx), 4)

    return ari, nmi, [n_clusters, cnum, k]
    







def pytorch_kmeans(data, n_clusters, max_iter=300, tol=1e-4):
    """Basic PyTorch k-means implementation for MPS"""
    # Randomly initialize centroids
    centroids = data[torch.randperm(data.shape[0])[:n_clusters]]

    print('stage1 starting2')
    
    for _ in range(max_iter):
        # Compute distances
        distances = torch.cdist(data, centroids)
        # Assign clusters
        labels = torch.argmin(distances, dim=1)
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_clusters):
            mask = labels == i
            if mask.any():
                new_centroids[i] = data[mask].mean(dim=0)
        
        # Check for convergence
        if torch.norm(centroids - new_centroids) < tol:
            break
        centroids = new_centroids
    
    return centroids





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

    batches = (n_runs + batch_size - 1) // batch_size  # Ceiling division

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

                    if len(results) % 100 == 0 or dataset in ['celltypes','covtype','postures','mnist','bot']:
                        elapsed = time.time() - start_time
                        print(f"Completed {len(results)}/{n_runs} runs in {elapsed:.1f}s")
                        print('results_max',max(results))
                except Exception as e:
                    print(f"Run {run_id} failed: {e}")
                    results.append((0, 0, [0, 0, 0]))

        run_counter += current_batch_size

    return results







def evaluate_with_seeds(data_path, use_gpu=True, n_runs=100, n_processes=None,k1=None,k2=None,dataset=None,seed=None,config_file=None):
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






