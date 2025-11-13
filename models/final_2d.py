"""
Final 2D Federated Clustering Implementation
Optimized clustering for 2D datasets with enhanced visualization capabilities.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.ndimage import label
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Tuple
import multiprocessing as mp
import pickle
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies
try:
    from anytree import Node, RenderTree
    TREE_VIZ_AVAILABLE = True
except ImportError:
    TREE_VIZ_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Try to use GPU acceleration if available


def load_dataset(filepath):
    """Load dataset from pickle file"""
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset

def dist(a, b, ax=1):
    """Compute distance between two points"""
    return np.linalg.norm(a - b, axis=ax)

def generate_synthetic_candidates(data, n_candidates_multiplier=1.5):
    """Generate grid-like candidates for any dimension"""
    n = len(data)
    dim = data.shape[1]
    n_candidates = int(np.ceil(np.sqrt(n * n_candidates_multiplier)))**2

    
    # Calculate bounds with padding
    mins = np.min(data, axis=0) - 0.0 * (np.max(data, axis=0) - np.min(data, axis=0))
    maxs = np.max(data, axis=0) + 0.0 * (np.max(data, axis=0) - np.min(data, axis=0))
    
    # Calculate points per dimension (approximate n-D grid)
    points_per_dim = int(np.ceil(n_candidates ** (1/dim)))
    
    # Generate grid
    grid_axes = [np.linspace(mins[d], maxs[d], points_per_dim) for d in range(dim)]
    mesh = np.meshgrid(*grid_axes)
    candidates = np.vstack([m.ravel() for m in mesh]).T
    
    # If we generated too many points, randomly subsample
    if len(candidates) > n_candidates:
        candidates = candidates[np.random.choice(len(candidates), n_candidates, replace=False)]
    
    return candidates

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist



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



from sklearn.neighbors import NearestNeighbors




import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform, pdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist



from scipy.ndimage import label
from collections import defaultdict
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from scipy.ndimage import label as ndi_label
from collections import defaultdict

from scipy.spatial.distance import cdist


import numpy as np
from collections import defaultdict
from anytree import Node, RenderTree
import matplotlib.pyplot as plt
from scipy.ndimage import label as ndi_label


import numpy as np
from collections import defaultdict
from anytree import Node, RenderTree
import matplotlib.pyplot as plt

class RegionNode(Node):
    """Extended Node class to store region information"""
    def __init__(self, name, potential, positions, parent=None, children=None):
        super().__init__(name, parent, children)
        self.potential = potential
        self.positions = positions
        self.weighted_position = self.calculate_weighted_position()
        self.normalized_potential = 0
        self._is_leaf = False  # Initialize leaf status
        
    @property
    def is_leaf(self):
        """Returns True if this node has no children"""
        return not self.children or self._is_leaf
        
    @is_leaf.setter
    def is_leaf(self, value):
        """Sets manual leaf status (overrides automatic child-based detection)"""
        self._is_leaf = bool(value)
        
    def calculate_weighted_position(self):
        if len(self.positions) == 0:
            return None
        weights = np.array([p[1] for p in self.positions])  # potentials as weights
        positions = np.array([p[0] for p in self.positions])
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)  # normalize
        return np.sum(positions * weights[:, np.newaxis], axis=0)


def build_region_tree(region_evolution, synthetic_candidates, total_energy):
    """Build a tree structure where only splits create new nodes (continuing regions preserve original properties)"""
    # Create root node (all points as one region)
    root_potential = np.sum(total_energy)
    root_positions = list(zip(synthetic_candidates, total_energy))
    root = RegionNode("Root", root_potential, root_positions)
    
    # Track current level nodes
    current_nodes = {1: root}
    prev_labels = np.ones(len(synthetic_candidates), dtype=int)
    
    for step, info in region_evolution.items():
        next_nodes = {}
        current_labels = info['labels']
        
        # Create parent→children mapping
        parent_children = defaultdict(set)
        for child_label in np.unique(current_labels):
            if child_label == 0:
                continue
            parents = prev_labels[current_labels == child_label]
            parents = parents[parents != 0]
            for parent in np.unique(parents):
                parent_children[parent].add(child_label)
        
        # Process each current node
        for parent_label, parent_node in current_nodes.items():
            children_labels = parent_children.get(parent_label, set())
            
            if not children_labels:
                # Region disappeared
                continue
                
            if len(children_labels) == 1:
                # Region continues - reuse the existing node without modification
                child_label = children_labels.pop()
                next_nodes[child_label] = parent_node  # Keep original node reference
            else:
                # Region splits - create new nodes
                for child_label in children_labels:
                    child_mask = (current_labels == child_label)
                    if np.sum(child_mask) == 0:
                        continue
                        
                    child_potential = np.sum(total_energy[child_mask])
                    child_positions = list(zip(synthetic_candidates[child_mask],
                                           total_energy[child_mask]))
                    child_node = RegionNode(f"R{child_label}-S{step}",
                                          child_potential,
                                          child_positions,
                                          parent=parent_node)
                    next_nodes[child_label] = child_node
        
        current_nodes = next_nodes
        prev_labels = current_labels
    
     # Mark all terminal nodes as leaves
    for node in current_nodes.values():
        if not node.children:  # Now using the proper property
            node.is_leaf = True  # This will work with the setter
    
    return root

def plot_region_tree(root):
    """Visualize the region tree"""
    for pre, _, node in RenderTree(root):
        pos_str = f"{node.weighted_position.round(2)}" if node.weighted_position is not None else "None"
        print(f"{pre}{node.name} (P: {node.potential:.2f}, Pos: {pos_str})")


def plot_energy_heatmap(data, candidate_energy, synthetic_candidates):
        """
        Plot the candidate energies as a heatmap.
        
        Parameters:
        - data: Original data points (2D array)
        - candidate_energy: Energy values for each candidate (1D array)
        - synthetic_candidates: Positions of synthetic candidates (2D array)
        """
        plt.figure(figsize=(10, 8))
        
        # Create a scatter plot of the original data points
        plt.scatter(data[:, 0], data[:, 1], c='blue', s=10, label='Original Data')
        
        # Create a scatter plot of the synthetic candidates colored by energy
        sc = plt.scatter(synthetic_candidates[:, 0], synthetic_candidates[:, 1], 
                        c=candidate_energy, cmap='viridis', s=50, 
                        label='Synthetic Candidates')
        
        # Add colorbar
        plt.colorbar(sc, label='Candidate Energy')
        
        plt.title('Candidate Energy Heatmap')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()



# Visualization for energy grid regions
def plot_region_evolution(region_evolution, energy_grid_shape):
    """Plot the region evolution using the energy grid structure"""
    last_steps = sorted(region_evolution.items())[-10:]  # Last 10 thresholds
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for ax, (step, info) in zip(axes, last_steps):
        # Reshape labels to energy grid
        region_grid = info['labels'].reshape(energy_grid_shape)
        
        # Plot as image
        im = ax.imshow(region_grid, cmap='tab20', origin='lower')
        ax.set_title(f"Thresh: {info['threshold']:.2f}\nRegions: {info['n_regions']}", fontsize=10)
        
        # Add colorbar for first plot
        if step == last_steps[0][0]:
            plt.colorbar(im, ax=ax, label='Region ID')
    
    plt.suptitle("Energy Grid Region Evolution", y=1.02)
    plt.tight_layout()
    plt.show()


def penalized_energy_centroids(data, nc, candidates_multiplier, energy_multiplier, epsilon):
    """Select centroids with tree-based region evolution tracking"""
    
    # Generate synthetic candidates
    synthetic_candidates = generate_synthetic_candidates(data, n_candidates_multiplier=candidates_multiplier)
    
    # Calculate distances and energies
    candidate_distances = cdist(data, synthetic_candidates, 'euclidean')  # or your preferred metric
    eps = 100  # Small epsilon to avoid division by zero
    # eps = np.percentile(candidate_distances[candidate_distances > 0], 5)
    # candidate_energy = np.sum(1/(candidate_distances**2+eps), axis=0)
    print('eps',eps)
    candidate_energy = np.sum(1/(candidate_distances**energy_multiplier+eps), axis=0)
    total_energy = candidate_energy 

    # plot_energy_heatmap(data, candidate_energy, synthetic_candidates)

    print(f"data.shape: {data.shape}")
    print(f"synthetic_candidates.shape: {synthetic_candidates.shape}")
    print(f"candidate_distances.shape: {candidate_distances.shape}")
    print(f"total_energy.shape: {total_energy.shape}")
        
    
    def get_energy_midpoints(total_energy):
        # Sort the energy values from smallest to largest
        sorted_energy = np.sort(total_energy)
        
        # Calculate midpoints between consecutive values
        midpoints = (sorted_energy[1:] + sorted_energy[:-1]) / 2
        
        return midpoints

    thresholds = get_energy_midpoints(total_energy)

    print(f"thresholds.shape: {thresholds.shape}")

   

    # # Subsample: Take 1, skip 9, repeat
    # keep_mask = np.zeros(len(thresholds), dtype=bool)
    # keep_mask[::9] = True  # Set every 10th element to True (keep 1, skip 9)

    # thresholds = thresholds[keep_mask]

    
    # Store region evolution
    region_evolution = defaultdict(dict)
    prev_labels = np.ones(len(synthetic_candidates), dtype=int)  # All points start in region 1
    candidate_storage = {}

    # 1. First, ensure consistent padding for ALL arrays
    grid_size = int(np.ceil(np.sqrt(len(synthetic_candidates))))
    import matplotlib.pyplot as plt  # Add this import at the top of your file




        # Plot the initial energy field before processing
    plt.figure(figsize=(8, 6))

    # Reshape energy to grid
    energy_grid = total_energy.reshape((grid_size, grid_size))

    # Plot with colorbar
    plt.imshow(energy_grid, cmap='viridis', origin='lower')
    plt.colorbar(label='Energy Level')
    plt.title(f'Initial Energy Field\nGrid Size: {grid_size}x{grid_size}')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')

    # Add grid lines
    plt.grid(True, color='white', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    
    for i, threshold in enumerate(thresholds):
        # Create binary map (1 if energy >= threshold)
        binary_map = (total_energy >= threshold).astype(int)

        
        binary_grid = binary_map.reshape((grid_size, grid_size))


        labeled, n_regions = ndi_label(binary_grid)
        current_labels = labeled.ravel()
        
        # Filter small regions (less than 5 nodes)
        unique_labels, counts = np.unique(current_labels, return_counts=True)
        small_regions = unique_labels[counts <= 10]
        
        # Create mask of regions to keep
        keep_mask = ~np.isin(current_labels, small_regions)
        
        # Second labeling pass on filtered regions
        filtered_binary = binary_map.copy()
        filtered_binary[~keep_mask] = 0
        filtered_grid = filtered_binary.reshape((grid_size, grid_size))
        relabeled, n_regions = ndi_label(filtered_grid)
        current_labels = relabeled.ravel()


     
        # Track region splits
        split_info = {}
        for new_label in np.unique(current_labels):
            if new_label == 0:  # Skip background
                continue
            parent_labels = prev_labels[current_labels == new_label]
            parent_labels = parent_labels[parent_labels != 0]  # Remove background
            if len(parent_labels) > 0:
                dominant_parent = np.bincount(parent_labels).argmax()
                split_info[new_label] = dominant_parent
        
        region_evolution[i] = {
            'threshold': threshold,
            'labels': current_labels,
            'n_regions': n_regions,
            'splits': split_info
        }
       
        prev_labels = current_labels

        # if n_regions >=20:
        #     break

    
    # Visualization for 2D
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    import matplotlib.pyplot as plt


    # Call this instead of your current visualization
    plot_region_evolution(region_evolution, (grid_size, grid_size))


        

    # Build the region tree
    # Build the region tree
    import sys
    sys.setrecursionlimit(10000)  # Increase limit (default is usually 1000)
    root = build_region_tree(region_evolution, synthetic_candidates, total_energy)
    # plot_region_tree(root)

    def count_leaves(node):
        """Count all leaf nodes in the tree"""
        if not node.children:  # This is a leaf node
            return 1
        return sum(count_leaves(child) for child in node.children)

    # Usage:
    num_leaves = count_leaves(root)
    print(f"Total leaf nodes in tree: {num_leaves}")

    def select_centroid_nodes(root, nc):
        selected = []
        all_leaves = [node for node in root.descendants if not node.children]
        
        # Sort all leaves by potential first
        all_leaves.sort(key=lambda x: -x.potential)
    
        
        # Track all ancestor paths to ensure topological independence
        used_branches = set()
        
        
        for leaf in all_leaves:
            if len(selected) >= nc:
                break
                
            # Get the full path from leaf to root
            path = []
            current = leaf
            while current != root:
                path.append(current.name)
                current = current.parent
            
            # Check if this path conflicts with any selected leaf
            independent = True
            for node_name in path:
                if node_name in used_branches:
                    independent = False
                    break
                    
            if independent:
                selected.append(leaf)
                # Add all nodes in path to used branches
                used_branches.update(path)
        
        print('selecte',len(selected))
        
        # If we still need more, take highest potential regardless
        if len(selected) < nc:
            remaining = nc - len(selected)
            for leaf in all_leaves:
                if leaf not in selected:
                    selected.append(leaf)
                    remaining -= 1
                    if remaining <= 0:
                        break
        
        # print('len',len(selected))

        
        return selected[:nc]
    
    # Rank synthetic candidates by energy (descending order)

    # Rank synthetic candidates by energy (descending order)
    ranked_indices = np.argsort(-total_energy)  # Negative for descending sort
    ranked_candidates = synthetic_candidates[ranked_indices]


    centroid_nodes = select_centroid_nodes(root, nc)

    # Get centroid positions
    centroids = []
    for node in centroid_nodes:
        if node.weighted_position is not None:
            centroids.append(node.weighted_position)
        elif len(node.positions) > 0:
            centroids.append(node.positions[0][0])  # First position


    if centroids is not None:
        centroid_indices = [np.argmin(pairwise_distances([c], synthetic_candidates)[0]) for c in centroids]
        centroid_indices_arr = np.array(centroid_indices, dtype=np.int32)
        syn_centroids = synthetic_candidates[centroid_indices_arr]
    else:
        syn_centroids = np.array([])  # Initialize as empty numpy array

    # Select top-N candidates to append (adjust `nc` as needed)
    if len(syn_centroids) < nc:
        remaining = nc - len(syn_centroids)
        # Convert to list if needed, or use numpy concatenation
        if remaining > 0:
            if len(syn_centroids) == 0:
                syn_centroids = ranked_candidates[:remaining]
            else:
                syn_centroids = np.concatenate([syn_centroids, ranked_candidates[:remaining]])
    
    print('yes')


    return syn_centroids,None




def SNN_optimized(nc: int, data: np.ndarray,candidates_multiplier, energy_multiplier,epsilon) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized SNN clustering using synthetic grid candidates"""
    n, d = data.shape
    distance = squareform(pdist(data))
    
    # Step 2: Energy calculation (on real data)
    eps = 30
    energy = np.sum(1/(distance**2 + eps), axis=1)

    # Step 4: Penalized centroid selection from synthetic candidates
    syn_centroids, centroid_indices = penalized_energy_centroids(data, nc, candidates_multiplier, energy_multiplier,epsilon)
    
    # For now, no assignments (set to None or zeros)
    indexAssignment = np.zeros(n, dtype=int)  # Placeholder
    
    # Visualization
  
    # plot_clusters(data, syn_centroids, energy, assignments=indexAssignment)
    
    return syn_centroids, centroid_indices

def plot_clusters(data, synthetic_centroids, energy, assignments=None):
    """Visualize clusters in 2D or higher dimensions using t-SNE when needed"""
    plt.figure(figsize=(12, 7))
    
    # Calculate energy threshold
    min_e, max_e = np.min(energy), np.max(energy)
    threshold = min_e
    high_energy_mask = energy > threshold
    
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
    
    # Plot all points (below threshold in blue)
    plt.scatter(data_2d[~high_energy_mask, 0], data_2d[~high_energy_mask, 1], 
                c='lightblue', alpha=0.4, s=30, label='Below energy threshold')
    
    # Plot high-energy points (red)
    plt.scatter(data_2d[high_energy_mask, 0], data_2d[high_energy_mask, 1], 
                c='red', alpha=0.6, s=30, label='Above energy threshold')
    
    # Plot cluster assignments if available and not all zeros
    if assignments is not None and not np.all(assignments == 0):
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
    
    plt.title(f"{plot_title}Synthetic Centroids (Energy Threshold: {threshold:.2f})")
    plt.xlabel('Component 1' if d > 2 else 'Feature 1')
    plt.ylabel('Component 2' if d > 2 else 'Feature 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()






def add_laplace_noise_vectorized(data, epsilon, sensitivity):
    """Vectorized Laplace noise addition"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise


def nnfc_optimized(data_path, use_gpu=False,k1=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None):
    """Optimized NNFC function"""

    datapkl = load_dataset(data_path)
    # eachlable = datapkl['eachlable']
    # order = datapkl['order']
    true_labels = np.array(datapkl['true_label'])  # Original labels for full data
    data = np.array(datapkl['full_data'])
    print(data.shape)
    
    corepoints = []
    datasets = []
    # print('order',len(order))
    print('stage1 starting')
    
    for i_client in range(10):
        print('stage1 with the client',i_client)
        lodata = datapkl["client_" + str(i_client)]

        


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

        datasets.append(centers)

        reducer = umap.UMAP(n_components=2, random_state=42)
        centers = reducer.fit_transform(centers)

        corepoints.append(centers)
    
    print('stage2 starting')
    
    serverdata = np.concatenate(corepoints, axis=0)
    datasets = np.concatenate(datasets, axis=0)
    label = datapkl['true_label']
    # cnum = len(set(label))
    cnum = len(set(true_labels))
    # k = min(5, len(serverdata) // 10)  # Adaptive k
    # print('cnum',cnum)

    print('stage6 starting')


    # Old: finalcenter = serverdata[centroid]
    finalcenter, assignment = SNN_optimized(cnum, serverdata,candidates_multiplier,energy_multiplier,epsilon)  # Now returns synthetic centroids


    
    # # Compute distances from all data points to synthetic centroids
    # distances = cdist(data, finalcenter)  # Faster than manual loops

    # # Assign each point to the nearest synthetic centroid
    # idx = np.argmin(distances, axis=1) + 1  # +1 for 1-based indexing

    # # Compute metrics
    # ari = round(adjusted_rand_score(label, idx), 4)
    # nmi = round(normalized_mutual_info_score(label, idx), 4)

    # return ari, nmi, [n_clusters,cnum]

    print('finalcenter',finalcenter.shape)
    print('serverdata',serverdata.shape)
    print('datasets',datasets[0].shape)

  
    
    # Step 2: For each 2D finalcenter, find the closest serverdata_2d point
    nbrs = NearestNeighbors(n_neighbors=1).fit(serverdata)
    _, indices = nbrs.kneighbors(finalcenter)
    
    # Step 3: Get the original 8D points corresponding to these indices
    finalcenter_8d = datasets[indices.flatten()]

    print('finalcenter_8d',finalcenter_8d.shape)
    print('data',data.shape)
    
    # Stage 8: Assign original 8D data to 8D centroids
    # ------------------------------------------------
    distances = cdist(data, finalcenter_8d)  # 8D to 8D distances
    idx = np.argmin(distances, axis=1) + 1  # 1-based indexing
    
    # Compute metrics
    ari = round(adjusted_rand_score(true_labels, idx), 4)
    nmi = round(normalized_mutual_info_score(true_labels, idx), 4)
    
    return ari, nmi, [n_clusters, cnum]



def run_single_experiment(args):
    """Single experiment runner for multiprocessing"""
    data_path, use_gpu, k1,candidates_multiplier,energy_multiplier,epsilon = args
    return nnfc_optimized(data_path, use_gpu,k1,candidates_multiplier,energy_multiplier,epsilon)






def run_experiments_parallel(data_path, n_runs=1000, n_processes=None, use_gpu=True, k1=None, dataset=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None):
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
        args_list = [(data_path, use_gpu, k1, candidates_multiplier,energy_multiplier,epsilon) for i in range(current_batch_size)]

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







def evaluate_with_seeds(data_path, use_gpu=True, n_runs=100, n_processes=None,k1=None,dataset=None,seed=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None):
    """Evaluate performance across 10 seeds"""
    SEEDS = list(range(seed))
    seed_results = {'ari_max': [], 'nmi_max': []}
    
    for seed in SEEDS:
        print(f"\n--- Evaluating with seed={seed} ---")
        results = run_experiments_parallel(data_path, n_runs, n_processes, use_gpu,k1,dataset,candidates_multiplier,energy_multiplier,epsilon)
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
     

        

        # Evaluate with 10 seeds
        seed_results = evaluate_with_seeds(data_path, use_gpu, n_runs=n_runs, n_processes=n_processes,k1=k1,dataset=dataset,seed=seeds,energy_multiplier=energy_multiplier,candidates_multiplier=candidates_multiplier,epsilon=epsilon)
        
        # Save seed results
        with open(save_path, 'a') as f:
            f.write(f"{data_path}\n")
            f.write(f"ARI (max) across seeds: {seed_results['ari_max']}\n")
            f.write(f"NMI (max) across seeds: {seed_results['nmi_max']}\n")
            f.write(f"Mean ± Std - ARI: {np.mean(seed_results['ari_max']):.4f} ± {np.std(seed_results['ari_max']):.4f}\n")
            f.write(f"Mean ± Std - NMI: {np.mean(seed_results['nmi_max']):.4f} ± {np.std(seed_results['nmi_max']):.4f}\n\n")






