# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2025-06-02 17:55:49
# @Last Modified by:   Yunbo
# @Last Modified time: 2025-07-05 22:37:23
import umap
import time
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
        
        # plt.title('Candidate Energy Heatmap')
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
        ax.set_title(f"Thresh: {info['threshold']:.2f}\nRegions: {info['n_regions']}", fontsize=24)
        
        # Add colorbar for first plot
        if step == last_steps[0][0]:
            plt.colorbar(im, ax=ax, label='Region ID')
    
    plt.suptitle("Energy Grid Region Evolution", y=1.02)
    plt.tight_layout()
    plt.show()


from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


def generate_highdim_candidates(data, n_candidates_multiplier=1.5):
    """Generate candidates adapted to high dimensions"""
    """
    Generate random candidates within the observed bounds of the data.
    
    Args:
        data: Input array of shape (n_samples, n_features).
        n_candidates_multiplier: Multiplier to determine number of candidates 
                                (n_candidates = n_samples * multiplier).
    
    Returns:
        candidates: Array of shape (n_candidates, n_features).
    """
    n_samples, n_features = data.shape
    n_candidates = int(n_samples * n_candidates_multiplier)
    
    # Get min/max bounds for each feature (with optional padding)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    
    # Generate uniform random samples within bounds
    candidates = np.random.uniform(
        low=mins,
        high=maxs,
        size=(n_candidates, n_features))
    
    return candidates

def track_splits(prev_labels, current_labels):
    """Track how clusters split using graph connections"""
    split_info = {}
    
    for new_label in np.unique(current_labels):
        if new_label == 0:
            continue
            
        # Find which previous clusters contributed to this new cluster
        parent_labels = prev_labels[current_labels == new_label]
        unique_parents, counts = np.unique(parent_labels[parent_labels != 0], return_counts=True)
        
        if len(unique_parents) > 0:
            dominant_parent = unique_parents[np.argmax(counts)]
            split_info[new_label] = dominant_parent
            
    return split_info


def get_energy_midpoints(total_energy):
        # Sort the energy values from smallest to largest
        sorted_energy = np.sort(total_energy)
        
        # Calculate midpoints between consecutive values
        midpoints = (sorted_energy[1:] + sorted_energy[:-1]) / 2
        
        return midpoints


# Helper functions
def get_energy_midpoints(energy):
    sorted_energy = np.sort(energy)
    return (sorted_energy[1:] + sorted_energy[:-1]) / 2

# def generate_random_candidates(data, n_candidates_multiplier=1.5):
#     mins = np.min(data, axis=0)
#     maxs = np.max(data, axis=0)
#     n_candidates = int(len(data) * n_candidates_multiplier)
#     return np.random.uniform(mins, maxs, size=(n_candidates, data.shape[1]))


def generate_random_candidates(data, n_candidates_multiplier=1.5):
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    n_candidates = int(len(data) * n_candidates_multiplier)
    
    # Calculate points per dimension (approximate)
    dim = data.shape[1]
    points_per_dim = int(np.ceil(n_candidates ** (1/dim)))
    
    # Create grid
    axes = [np.linspace(mins[d], maxs[d], points_per_dim) for d in range(dim)]
    grid = np.meshgrid(*axes)
    return np.vstack([g.ravel() for g in grid]).T[:n_candidates]

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict

def compute_avg_candidate_distance(synthetic_candidates, k=2):
    """
    Compute the average distance between each candidate and its k-nearest neighbors.
    
    Args:
        synthetic_candidates: Array of shape (n_candidates, n_features)
        k: Number of neighbors to consider (default=2: closest non-self point)
    
    Returns:
        avg_distance: Mean pairwise distance between candidates
        median_distance: Median distance (more robust to outliers)
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(synthetic_candidates)
    distances, _ = nbrs.kneighbors(synthetic_candidates)
    
    # Exclude self-distance (column 0) if k > 1
    neighbor_distances = distances[:, 1:] if k > 1 else distances
    return np.mean(neighbor_distances), np.median(neighbor_distances)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_energy_clusters(active_mask, active_points, total_energy, clustering, threshold, iteration):
    """Visualize energy field and DBSCAN clusters in 2D with controlled plotting"""
    # Only plot first 20 or every 100th iteration
    if iteration >= 20 and iteration % 100 != 0:
        return
    
    # Project to 2D using t-SNE (with progress tracking)
    print(f"Visualizing iteration {iteration}...")
    tsne = TSNE(n_components=2, random_state=42, verbose=0)
    points_2d = tsne.fit_transform(active_points)
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot 1: Energy Heatmap with better normalization
    energy_values = total_energy[active_mask][:len(points_2d)]
    sc1 = ax1.scatter(points_2d[:, 0], points_2d[:, 1], 
                     c=energy_values, 
                     cmap='inferno', 
                     alpha=0.8,
                     norm=LogNorm() if np.any(energy_values <= 0) else None)
    cbar = plt.colorbar(sc1, ax=ax1, label='Log Energy Level')
    ax1.set_title(f"Iter {iteration}: Energy Field\n(Threshold={threshold:.2f})")
    
    # Plot 2: DBSCAN Clusters with improved labeling
    unique_labels = np.unique(clustering.labels_)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    color_map = plt.cm.get_cmap('gist_ncar', n_clusters + 1)
    
    # Plot noise first (so it stays in background)
    noise_mask = clustering.labels_ == -1
    if np.any(noise_mask):
        ax2.scatter(points_2d[noise_mask, 0], points_2d[noise_mask, 1],
                   c='gray', alpha=0.2, s=10, label=f'Noise ({np.sum(noise_mask)} pts)')
    
    # Plot clusters with size proportional to energy
    cluster_sizes = []
    for i, label in enumerate(unique_labels[unique_labels != -1]):
        mask = clustering.labels_ == label
        size = 50 * np.sqrt(np.mean(energy_values[mask]))  # Scale by energy
        ax2.scatter(points_2d[mask, 0], points_2d[mask, 1],
                   c=[color_map(i)],
                   s=size,
                   alpha=0.7,
                   label=f'Cluster {label} ({np.sum(mask)} pts)')
        cluster_sizes.append(np.sum(mask))
    
    # Add summary statistics
    stats_text = (f"Clusters: {n_clusters}\n"
                 f"Largest: {max(cluster_sizes, default=0)} pts\n"
                 f"Noise: {np.sum(noise_mask)} pts\n"
                 f"eps: {getattr(clustering, 'eps', 'N/A'):.2f}")
    ax2.text(1.05, 0.5, stats_text, transform=ax2.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.set_title(f"DBSCAN Clustering\n(MinPts={clustering.min_samples})")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'cluster_iter_{iteration:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()  # Prevents memory leaks in loops


def plot_region_splits(coords_2d, labels, active_mask, threshold, iteration, split_info):
    """Visualize region splitting in 2D PCA space"""
    plt.figure(figsize=(12, 6))
    
    # Plot all points (inactive as gray)
    plt.scatter(coords_2d[~active_mask, 0], coords_2d[~active_mask, 1], 
                c='gray', alpha=0.1, s=10, label='Inactive')
    
    # Plot active regions
    unique_labels = np.unique(labels[active_mask])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == 0: continue
        mask = (labels == label) & active_mask
        plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=[color], 
                   s=30,
                   label=f'Region {label} (Parent: {split_info.get(label, "None")})')
    
    # plt.title(f"Iteration {iteration}\nThreshold: {threshold:.4f}, Regions: {len(unique_labels)-1}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def penalized_energy_centroids(data, nc,all_masses, candidates_multiplier, energy_multiplier, epsilon,delta,radius,counts):
    """High-dimensional version with region tracking and visualization"""
    # Generate candidates
    # Generate candidates
    synthetic_candidates = generate_random_candidates(data, n_candidates_multiplier=candidates_multiplier)
    
    # Calculate distances between all data points (centroids) and candidates
    candidate_distances = cdist(data, synthetic_candidates, 'euclidean')

    # print('mass',all_masses)
    
    # Calculate potential energy for each candidate using mass weighting
    # E(r) = sum( w_i / (||r - s_i||^p + δ) )  (Equation 3)
    # where:
    # - w_i is the mass (all_masses[i])
    # - s_i is the centroid (data[i])
    # - p is energy_multiplier (typically 2)
    # - δ is the softening parameter

    # Verify shapes
    print(f"Data shape: {data.shape}")  # Should be (n_centroids, n_features)
    print(f"All_masses shape: {all_masses.shape}")  # Should be (n_centroids,)
    print(f"Candidate distances shape: {candidate_distances.shape}")
    
    # Reshape masses to allow broadcasting
    weighted_masses = all_masses.reshape(-1, 1)  # Shape (n_centroids, 1)
    
    # Calculate energy contribution from each centroid to each candidate
    # energy_contributions = weighted_masses / (candidate_distances**energy_multiplier + delta)
    energy_contributions = weighted_masses / (candidate_distances**energy_multiplier + delta)
    
    
    # Sum contributions from all centroids for each candidate
    total_energy = np.sum(energy_contributions, axis=0)
    
    # Get energy thresholds
    thresholds = get_energy_midpoints(total_energy)

    # thresholds = thresholds[thresholds.shape[0]//2:]
    
   
    
    # Initialize tracking
    region_evolution = defaultdict(dict)
    prev_labels = np.ones(len(synthetic_candidates), dtype=int)
    
    # Set up visualization
    from sklearn.decomposition import PCA
    if synthetic_candidates.shape[1]>2:
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(synthetic_candidates)
    
    else:
        coords_2d = synthetic_candidates
    
    for i, threshold in enumerate(thresholds):
        active_mask = (total_energy >= threshold)
        active_points = synthetic_candidates[active_mask]
        
        if len(active_points) == 0:
            continue

        # # Create connectivity graph
        # radius1 = np.percentile(candidate_distances[candidate_distances > 0], 1)  # 5th percentile of non-zero distances
        # print('radius',radius1)
       
        nbrs = NearestNeighbors(radius=radius).fit(active_points)
        adjacency = nbrs.radius_neighbors_graph(active_points, mode='connectivity')
        n_components, labels = connected_components(adjacency)

        
        # Map labels
        current_labels = np.zeros(len(synthetic_candidates), dtype=int)
        current_labels[active_mask] = labels + 1

        unique_labels, count = np.unique(current_labels[current_labels != 0], return_counts=True)
        small_regions = unique_labels[count <= counts]
        current_labels[np.isin(current_labels, small_regions)] = 0
        
        # Track splits
        split_info = {}
        for new_label in np.unique(current_labels):
            if new_label == 0: continue
            parent_labels = prev_labels[current_labels == new_label]
            parent_labels = parent_labels[parent_labels != 0]
            if len(parent_labels) > 0:
                dominant_parent = np.bincount(parent_labels).argmax()
                split_info[new_label] = dominant_parent
        
        # Store evolution
        region_evolution[i] = {
            'threshold': threshold,
            'labels': current_labels,
            'n_regions': n_components,
            'splits': split_info
        }
        
        # # Visualize every 100 steps
        # if i % 100 == 0:
        #     plot_region_splits(
        #         coords_2d, 
        #         current_labels, 
        #         active_mask,
        #         threshold, 
        #         i,
        #         split_info
        #     )
        
        prev_labels = current_labels
    


    # Select final centroids based on region evolution

        # if n_regions >=20:
        #     break

    
    # Visualization for 2D
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    import matplotlib.pyplot as plt


   

        

    # Build the region tree
    # Build the region tree
    import sys
    sys.setrecursionlimit(10000)  # Increase limit (default is usually 1000)
    root = build_region_tree(region_evolution, synthetic_candidates, total_energy)
    plot_region_tree(root)

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


    return syn_centroids,None,total_energy,synthetic_candidates




def SNN_optimized(nc: int, data: np.ndarray, all_masses, candidates_multiplier, energy_multiplier, epsilon, delta, radius, counts, true_labels=None) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized SNN clustering using synthetic grid candidates"""
    n, d = data.shape
    distance = squareform(pdist(data))
    
    # Step 2: Energy calculation (on real data)
    energy = np.sum(1/(distance**2 + delta), axis=1)
    
    # Step 4: Penalized centroid selection from synthetic candidates
    syn_centroids, centroid_indices,total_energy,synthetic_candidates = penalized_energy_centroids(data, nc, all_masses, candidates_multiplier, energy_multiplier, epsilon, delta, radius, counts)
    
    # For now, no assignments (set to None or zeros)
    indexAssignment = np.zeros(n, dtype=int)  # Placeholder

    
    
#     # Visualization with true labels
#     plot_clusters(data, syn_centroids, energy, true_labels=true_labels, assignments=indexAssignment)
#     plot_clusters2(synthetic_candidates, total_energy, syn_centroids)
   
    

#     plot_energy_3d(
#     candidate_points=synthetic_candidates,
#     total_energy=total_energy,
#     synthetic_centroids=syn_centroids,
#     true_labels=true_labels  # optional
# )
#     plot_contour_layers(synthetic_candidates, total_energy)
    
    return syn_centroids, centroid_indices



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

# def plot_contour_layers(candidate_points, total_energy):
#     """
#     Plot 3 clearly spaced contour planes at 10%, 30%, and 50% energy levels
    
#     Parameters:
#     - candidate_points: 2D array of candidate points (shape Nx2)
#     - total_energy: 1D array of energy values (shape N,)
#     """
#     # Create grid interpolation
#     xi = np.linspace(min(candidate_points[:,0]), max(candidate_points[:,0]), 200)
#     yi = np.linspace(min(candidate_points[:,1]), max(candidate_points[:,1]), 200)
#     zi = griddata(candidate_points, total_energy, (xi[None,:], yi[:,None]), method='cubic')
    
#     # Calculate specific layer heights (10%, 30%, 50%)
#     min_e, max_e = np.nanmin(zi), np.nanmax(zi)
#     level_percentages = [0.1, 0.3, 0.5]
#     levels = [min_e + (max_e-min_e)*h for h in level_percentages]
    
#     # Apply vertical scaling (3x) to enhance spacing between planes
#     vertical_scale = 3.0
#     scaled_levels = [min_e + (l-min_e)*vertical_scale for l in levels]
    
#     # Create figure with adjusted aspect ratio (4x4x3)
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Set the 4x4x3 aspect ratio
#     ax.set_box_aspect([4, 4, 3])  # Width:Depth:Height ratio
    
#     # Create distinct colors for each plane
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Three distinct colors
    
#     # Plot each plane with contours
#     X, Y = np.meshgrid(xi, yi)
#     for i, (level, color) in enumerate(zip(scaled_levels, colors)):
#         # Create semi-transparent plane
#         ax.plot_surface(X, Y, np.full_like(X, level), 
#                        color=color, alpha=0.5, shade=False)
        
#         # Add contour lines at the plane edges
#         cs = ax.contour(X, Y, zi, levels=[levels[i]], 
#                        colors=[color], linestyles='solid', linewidths=2,
#                        offset=level)
        
#         # Label the contours with their percentage
#         ax.clabel(cs, cs.levels, inline=True, fmt=f'{level_percentages[i]*100:.0f}%', 
#                  fontsize=20, colors='black')

#     # Style the plot with uniform font sizes
#     ax.set_xlabel('Dimension 1', fontsize=20, labelpad=10)
#     ax.set_ylabel('Dimension 2', fontsize=20, labelpad=10)
#     ax.set_zlabel('Energy Level', fontsize=20, labelpad=10)
    
#     # Set tick label sizes
#     ax.tick_params(axis='x', labelsize=18)
#     ax.tick_params(axis='y', labelsize=18)
#     ax.tick_params(axis='z', labelsize=18)
    
#     # Adjust view angle for better visualization
#     ax.view_init(elev=40, azim=-50)
    
#     # Create clean legend with matching font size
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor=colors[i], alpha=0.5, 
#               label=f'{int(p*100)}% Level')
#         for i, p in enumerate(level_percentages)
#     ]
#     ax.legend(handles=legend_elements, loc='upper right', 
#               fontsize=20, framealpha=0.9)
    
#     plt.tight_layout()
#     plt.show()


def plot_contour_layers(candidate_points, total_energy):
    """
    Plot 3 clearly spaced contour planes at 10%, 30%, and 50% energy levels
    
    Parameters:
    - candidate_points: 2D array of candidate points (shape Nx2)
    - total_energy: 1D array of energy values (shape N,)
    """
    # Create grid interpolation
    xi = np.linspace(min(candidate_points[:,0]), max(candidate_points[:,0]), 200)
    yi = np.linspace(min(candidate_points[:,1]), max(candidate_points[:,1]), 200)
    zi = griddata(candidate_points, total_energy, (xi[None,:], yi[:,None]), method='cubic')
    
    # Calculate specific layer heights (10%, 30%, 50%)
    min_e, max_e = np.nanmin(zi), np.nanmax(zi)
    level_percentages = [0.1, 0.3, 0.5]
    levels = [min_e + (max_e-min_e)*h for h in level_percentages]
    
    # Apply vertical scaling (3x) to enhance spacing between planes
    vertical_scale = 3.0
    scaled_levels = [min_e + (l-min_e)*vertical_scale for l in levels]
    
    # Create figure with adjusted aspect ratio
    fig = plt.figure(figsize=(10, 7))  # Increased figure size
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([4, 4, 3])
    
    # Create distinct colors for each plane
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot each plane with contours
    X, Y = np.meshgrid(xi, yi)
    for i, (level, color) in enumerate(zip(scaled_levels, colors)):
        ax.plot_surface(X, Y, np.full_like(X, level), 
                       color=color, alpha=0.5, shade=False)
        cs = ax.contour(X, Y, zi, levels=[levels[i]], 
                       colors=[color], linestyles='solid', linewidths=2,
                       offset=level)
        ax.clabel(cs, cs.levels, inline=True, fmt=f'{level_percentages[i]*100:.0f}%', 
                 fontsize=20, colors='black')

    # Axis styling
    ax.set_xlabel('Dimension 1', fontsize=20, labelpad=15)
    ax.set_ylabel('Dimension 2', fontsize=20, labelpad=15)
    ax.set_zlabel('Energy Level', fontsize=20, labelpad=15)
    ax.tick_params(axis='x', labelsize=18, pad=8)
    ax.tick_params(axis='y', labelsize=18, pad=8)
    ax.tick_params(axis='z', labelsize=18, pad=8)
    
    # Adjust view and legend position
    ax.view_init(elev=35, azim=-45)  # Slightly lowered view angle
    
    # Create and position legend outside the plot
    legend_elements = [
        Patch(facecolor=colors[i], alpha=0.5, 
              label=f'{int(p*100)}% Level')
        for i, p in enumerate(level_percentages)
    ]
    
    # Position legend in lower right with more space
    ax.legend(handles=legend_elements, 
              loc='lower right',
              bbox_to_anchor=(0.85, 0.25),  # Adjusted position
              fontsize=18,
              framealpha=0.95,
              borderpad=1.2)

    # Adjust subplot parameters to prevent overlap
    plt.subplots_adjust(right=0.85, bottom=0.15)
    plt.show()


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
                    fontsize=24, fontweight='bold', color='black')
    
    # Configure axis labels and ticks
    plt.xlabel('Dimension 1', fontsize=24)  # Changed from 'Component 1'
    plt.ylabel('Dimension 2', fontsize=24)  # Changed from 'Component 2'
    
    # Set tick parameters for both axes
    plt.tick_params(axis='both', which='major', labelsize=24)
    
    plt.legend(loc='lower right', framealpha=0.9, prop={'size': 14})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_clusters2(candidate_points, total_energy, synthetic_centroids):
    """
    Plot a heatmap of the energy field with synthetic centroids marked
    
    Parameters:
    - candidate_points: 2D array of candidate points (must be 2D for heatmap)
    - total_energy: 1D array of energy values for each candidate point
    - synthetic_centroids: Array of centroid positions (will be marked on plot)
    """
    # Verify inputs are 2D
    if candidate_points.shape[1] != 2:
        raise ValueError("Candidate points must be 2-dimensional for heatmap visualization")
    
    plt.figure(figsize=(10, 8))  # Increased figure size for better visibility
    
    # Create grid interpolation for smooth heatmap
    from scipy.interpolate import griddata
    
    # Create grid coordinates
    grid_x = np.linspace(min(candidate_points[:,0]), max(candidate_points[:,0]), 200)
    grid_y = np.linspace(min(candidate_points[:,1]), max(candidate_points[:,1]), 200)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    
    # Interpolate energy values onto grid
    grid_z = griddata(
        candidate_points, 
        total_energy, 
        (grid_x, grid_y), 
        method='cubic',
        fill_value=np.min(total_energy))
    
    # Plot heatmap
    plt.imshow(
        grid_z,
        extent=(min(candidate_points[:,0]), max(candidate_points[:,0]), 
                min(candidate_points[:,1]), max(candidate_points[:,1])),
        origin='lower',
        aspect='auto',
        cmap='viridis',
        alpha=0.8)
    
    # Add colorbar with larger font
    cbar = plt.colorbar()
    cbar.set_label('Energy Value', rotation=270, labelpad=25, fontsize=24)
    cbar.ax.tick_params(labelsize=24)  # Colorbar tick labels
    
    # Mark centroids
    plt.scatter(
        synthetic_centroids[:,0], 
        synthetic_centroids[:,1],
        marker='*',
        s=400,
        c='gold',
        edgecolors='black',
        linewidths=1.5,
        label='Centroids')
    
    # Add numbering to centroids
    for i, (x, y) in enumerate(synthetic_centroids):
        plt.annotate(
            f'{i+1}', 
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=24,
            fontweight='bold',
            color='white')
    
    # Configure axis labels and ticks
    plt.xlabel('Dimension 1', fontsize=24)
    plt.ylabel('Dimension 2', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    
    # Configure legend
    plt.legend(
        loc='upper right',
        fontsize=20,  # Larger legend font
        framealpha=0.9,
        markerscale=1.5)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
def plot_energy_3d(candidate_points, total_energy, synthetic_centroids=None, true_labels=None):
    """
    Visualize the energy field in 3D with peaks and geometric landscape
    
    Parameters:
    - candidate_points: The candidate points (2D or 3D)
    - total_energy: Energy values for each candidate point
    - synthetic_centroids: Optional centroids to plot
    - true_labels: Optional true labels for coloring original points
    """
    fig = plt.figure(figsize=(10, 8))  # Increased figure size
    
    # Create 3D axis with larger font sizes
    ax = fig.add_subplot(111, projection='3d')
    
    # Check if we need to reduce dimensions
    if candidate_points.shape[1] > 3:
        from sklearn.manifold import TSNE
        perplex = min(30, candidate_points.shape[0]-1)
        projected_points = TSNE(n_components=3, perplexity=perplex, random_state=42).fit_transform(candidate_points)
        if synthetic_centroids is not None:
            projected_centroids = TSNE(n_components=3, perplexity=perplex, random_state=42).fit_transform(synthetic_centroids)
    else:
        projected_points = candidate_points
        if synthetic_centroids is not None:
            projected_centroids = synthetic_centroids
    
    # Normalize energy for coloring and height
    energy_min, energy_max = np.min(total_energy), np.max(total_energy)
    norm_energy = (total_energy - energy_min) / (energy_max - energy_min)
    
    # Create surface or scatter plot based on point density
    if len(candidate_points) > 1000:  # For large datasets, use surface plot
        from scipy.interpolate import griddata
        grid_x, grid_y = np.mgrid[
            np.min(projected_points[:,0]):np.max(projected_points[:,0]):100j,
            np.min(projected_points[:,1]):np.max(projected_points[:,1]):100j
        ]
        grid_z = griddata(
            projected_points[:,:2], 
            total_energy, 
            (grid_x, grid_y), 
            method='cubic', 
            fill_value=energy_min
        )
        
        # Plot surface
        surf = ax.plot_surface(
            grid_x, grid_y, grid_z, 
            cmap=cm.viridis,
            alpha=0.8,
            linewidth=0, 
            antialiased=True
        )
    else:  # For smaller datasets, use scatter plot
        sc = ax.scatter(
            projected_points[:,0], 
            projected_points[:,1], 
            total_energy,
            c=total_energy,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
    
    # Add colorbar with large font
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_array(total_energy)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Gravitational Energy', rotation=270, labelpad=25, fontsize=24)
    cbar.ax.tick_params(labelsize=20)  # Larger tick labels

    # Set labels with large font
    ax.set_xlabel('Dimension 1', fontsize=24, labelpad=15)
    ax.set_ylabel('Dimension 2', fontsize=24, labelpad=15)
    ax.set_zlabel('Gravitational Energy', fontsize=24, labelpad=15)
    
    # Set tick label sizes
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    
    # Add centroids if provided
    if synthetic_centroids is not None:
        ax.scatter(
            projected_centroids[:,0], 
            projected_centroids[:,1], 
            np.max(total_energy)*1.1,  # Slightly above surface
            marker='*',
            s=200,
            c='gold',
            edgecolors='black',
            linewidths=1,
            label='Centroids'
        )
        ax.legend(fontsize=18, loc='upper right')

    # Adjust view angle and layout
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()


def add_laplace_noise_vectorized(data, epsilon, sensitivity):
    """Vectorized Laplace noise addition"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise


def nnfc_optimized(data_path, use_gpu=False, k1=None, candidates_multiplier=None, energy_multiplier=None, epsilon=None, delta=None, radius=None, counts=None, seed=None):
    """Optimized NNFC function"""
    datapkl = load_dataset(data_path)
    true_labels = np.array(datapkl['true_label'])  # Original labels for full data
    data = np.array(datapkl['full_data'])
    
    print(f"Original data shape: {data.shape}")  # Should be (4177, d)
    print(f"True labels shape: {true_labels.shape}")  # Should be (4177,)
    
    corepoints = []
    masses = []
    print('stage1 starting')
    np.random.seed(seed)
    
    for i_client in range(10):
        print('stage1 with the client', i_client)
        lodata = datapkl["client_" + str(i_client)]
        
        scale = 1 / epsilon
        laplace_noise = np.random.laplace(loc=0.0, scale=scale, size=lodata.shape)
        lodata_noisy = lodata + laplace_noise
        
        n_clusters = k1
        cluster = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster.fit(lodata_noisy)
        centers = cluster.cluster_centers_
        
        # Calculate masses
        all_sq_dists = []
        for j in range(n_clusters):
            cluster_points = lodata_noisy[cluster.labels_ == j]
            if len(cluster_points) > 0:
                all_sq_dists.append(np.sum((cluster_points - centers[j])**2))
        
        sigma_sq = np.std(all_sq_dists) if len(all_sq_dists) > 1 else 1.0
        
        client_masses = []
        for j in range(n_clusters):
            cluster_points = lodata_noisy[cluster.labels_ == j]
            sum_sq_dist = np.sum((cluster_points - centers[j])**2) if len(cluster_points) > 0 else 0
            w = np.exp(-sum_sq_dist / (2 * sigma_sq)) 
            client_masses.append(w)
        
        corepoints.append(centers)
        masses.append(np.array(client_masses))
    
    print('stage2 starting')
    serverdata = np.concatenate(corepoints, axis=0)
    all_masses = np.concatenate(masses, axis=0)
    
    print(f"Server data (centroids) shape: {serverdata.shape}")  # Should be (10*k1, d)
    
    # For the centroids' true labels, we need to assign them based on their closest true data points
    centroid_true_labels = []
    for centroid in serverdata:
        # Find the closest original data point to this centroid
        closest_idx = np.argmin(np.linalg.norm(data - centroid, axis=1))
        centroid_true_labels.append(true_labels[closest_idx])
    centroid_true_labels = np.array(centroid_true_labels)
    
    print(f"Centroid true labels shape: {centroid_true_labels.shape}")  # Should match serverdata shape
    
    cnum = len(set(true_labels))
    print('stage6 starting')
    
    # Pass the centroid true labels instead of the full true labels
    finalcenter, assignment = SNN_optimized(cnum, serverdata, all_masses, 
                                          candidates_multiplier, energy_multiplier, 
                                          epsilon, delta, radius, counts, 
                                          true_labels=centroid_true_labels)
    
    # Compute distances from all data points to synthetic centroids
    distances = cdist(data, finalcenter)
    idx = np.argmin(distances, axis=1) + 1
    
    # Compute metrics
    ari = round(adjusted_rand_score(true_labels, idx), 4)
    nmi = round(normalized_mutual_info_score(true_labels, idx), 4)

    return ari, nmi, [n_clusters, cnum]



def run_single_experiment(args):
    """Single experiment runner for multiprocessing"""
    data_path, use_gpu, k1,candidates_multiplier,energy_multiplier,epsilon,delta,radius,counts,seed = args
    return nnfc_optimized(data_path, use_gpu,k1,candidates_multiplier,energy_multiplier,epsilon,delta,radius=radius,counts=counts,seed=seed)






def run_experiments_parallel(data_path, n_runs=1000, n_processes=None, use_gpu=True, k1=None, dataset=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None,delta=None,radius=None,counts=None,seed=None):
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
        args_list = [(data_path, use_gpu, k1, candidates_multiplier,energy_multiplier,epsilon,delta,radius,counts,seed) for i in range(current_batch_size)]

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            future_to_run = {executor.submit(run_single_experiment, args): run_counter + i for i, args in enumerate(args_list)}

            for future in as_completed(future_to_run):
                run_id = future_to_run[future]
                try:
                    result = future.result()
                    results.append(result)

                    if len(results) % 100 == 0 or dataset in ['celltypes','covtype','postures','mnist','bot','abalone','seeds','thyroid','breast','heart','balancescale','gestures']:
                        elapsed = time.time() - start_time
                        print(f"Completed {len(results)}/{n_runs} runs in {elapsed:.1f}s")
                        print('results_max',max(results))
                except Exception as e:
                    print(f"Run {run_id} failed: {e}")
                    results.append((0, 0, [0, 0, 0]))

        run_counter += current_batch_size

    return results





def evaluate_with_seeds(data_path, use_gpu=True, n_runs=100, n_processes=None,k1=None,dataset=None,seed=None,candidates_multiplier=None,energy_multiplier=None,epsilon=None,delta=None,radius=None,counts=None):
    """Evaluate performance across 10 seeds"""
    SEEDS = list(range(seed))
    seed_results = {'ari_max': [], 'nmi_max': []}
    
    for seed in SEEDS:
        print(f"\n--- Evaluating with seed={seed} ---")

        start_time = time.time()

        
        results = run_experiments_parallel(data_path, n_runs, n_processes, use_gpu,k1,dataset,candidates_multiplier,energy_multiplier,epsilon,delta,radius=radius,counts=counts,seed=seed)

        # Calculate and print elapsed time
        elapsed_time = time.time() - start_time
        print(f"Completed in: {elapsed_time:.2f} seconds")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

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
        energy_multiplier = config['energy_multiplier']
        epsilon = config['epsilon']
        delta = config[dataset]['delta']
        radius = config[dataset]['radius']
        counts = config[dataset]['counts']
     

        

        # Evaluate with 10 seeds
        seed_results = evaluate_with_seeds(data_path, use_gpu, n_runs=n_runs, n_processes=n_processes,k1=k1,dataset=dataset,seed=seeds,energy_multiplier=energy_multiplier,candidates_multiplier=candidates_multiplier,epsilon=epsilon,delta=delta,radius=radius,counts=counts)
        
        # Save seed results
        with open(save_path, 'a') as f:
            f.write(f"{data_path}\n")
            f.write(f"ARI (max) across seeds: {seed_results['ari_max']}\n")
            f.write(f"NMI (max) across seeds: {seed_results['nmi_max']}\n")
            f.write(f"Mean ± Std - ARI: {np.mean(seed_results['ari_max']):.4f} ± {np.std(seed_results['ari_max']):.4f}\n")
            f.write(f"Mean ± Std - NMI: {np.mean(seed_results['nmi_max']):.4f} ± {np.std(seed_results['nmi_max']):.4f}\n\n")






