"""
Constrained Federated Clustering (CFC) Implementation
A federated clustering algorithm with differential privacy and synthetic centroid generation.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cdist
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
import pickle
import time
import os
import warnings
warnings.filterwarnings('ignore')


def load_dataset(filepath):
    """Load dataset from pickle file"""
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset


def generate_synthetic_candidates(data, n_candidates_multiplier=1.5):
    """Generate synthetic candidate points using grid-based approach"""
    n_candidates = int(len(data) * n_candidates_multiplier)
    
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    
    if data.shape[1] == 2:
        grid_size = int(np.sqrt(n_candidates))
        x = np.linspace(mins[0], maxs[0], grid_size)
        y = np.linspace(mins[1], maxs[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
    else:
        candidates = np.random.uniform(mins, maxs, (n_candidates, data.shape[1]))
    
    return candidates


def penalized_energy_centroids(data, nc, candidates_multiplier, energy_multiplier):
    """Select centroids from synthetic candidates using energy-based method"""
    synthetic_candidates = generate_synthetic_candidates(data, n_candidates_multiplier=candidates_multiplier)
    
    candidate_distances = cdist(data, synthetic_candidates, 'euclidean')
    eps = 5
    candidate_energy = np.sum(1/(candidate_distances**energy_multiplier + eps), axis=0)
    
    # Apply density-aware weighting
    log_energy = np.log(candidate_energy)
    candidate_weights = np.exp(log_energy - np.max(log_energy))
    total_energy = candidate_energy

    # Visualization for 2D data
    if synthetic_candidates.shape[1] == 2:
        x = synthetic_candidates[:, 0]
        y = synthetic_candidates[:, 1]
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x, y, c=total_energy, cmap='viridis', s=40)
        plt.colorbar(scatter, label='Candidate Energy')
        plt.title("Energy Map of Synthetic Candidates")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Select centroids with highest energy
    centroid_indices = []
    for _ in range(nc):
        scores = total_energy.copy()
        scores[centroid_indices] = -np.inf
        next_idx = np.argmax(scores)
        centroid_indices.append(int(next_idx))
    
    centroid_indices_arr = np.array(centroid_indices, dtype=np.int32)
    return synthetic_candidates[centroid_indices_arr], centroid_indices_arr


def SNN_optimized(nc, data, candidates_multiplier, energy_multiplier):
    """Optimized SNN clustering using synthetic grid candidates"""
    n, d = data.shape
    
    # Generate synthetic centroids using energy-based method
    syn_centroids, centroid_indices = penalized_energy_centroids(
        data, nc, candidates_multiplier, energy_multiplier
    )
    
    indexAssignment = np.zeros(n, dtype=int)
    
    if d <= 3:
        plot_clusters(data, syn_centroids, np.ones(n), assignments=indexAssignment)
    
    return syn_centroids, centroid_indices


def plot_clusters(data, synthetic_centroids, energy, assignments=None):
    """Visualize clustering results"""
    plt.figure(figsize=(12, 7))
    
    min_e, max_e = np.min(energy), np.max(energy)
    threshold = min_e
    high_energy_mask = energy > threshold
    
    plt.scatter(data[~high_energy_mask, 0], data[~high_energy_mask, 1], 
                c='lightblue', alpha=0.4, s=30, label='Below energy threshold')
    
    plt.scatter(data[high_energy_mask, 0], data[high_energy_mask, 1], 
                c='red', alpha=0.6, s=30, label='Above energy threshold')
    
    if assignments is not None and not np.all(assignments == 0):
        scatter = plt.scatter(data[:, 0], data[:, 1], c=assignments, 
                             cmap='tab20', alpha=0.7, s=30, label='Cluster assignments')
        plt.colorbar(scatter, label='Cluster ID')
    
    plt.scatter(synthetic_centroids[:, 0], synthetic_centroids[:, 1],
                marker='*', s=400, c='gold', edgecolors='black',
                linewidths=2, label='Synthetic Centroids', zorder=5)
    
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
    """Add Laplace noise for differential privacy"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise


def nnfc_optimized(data_path, use_gpu=False, k1=None, candidates_multiplier=None, 
                   energy_multiplier=None, epsilon=None):
    """Main CFC clustering function"""
    datapkl = load_dataset(data_path)
    true_labels = np.array(datapkl['true_label'])
    data = np.array(datapkl['full_data'])
    print(f"Data shape: {data.shape}")
    
    corepoints = []
    print('Stage 1: Local clustering starting')
    
    # Stage 1: Local clustering on each client
    for i_client in range(10):
        print(f'Processing client {i_client}')
        lodata = datapkl[f"client_{i_client}"]
        
        # Add differential privacy noise
        scale = 1 / epsilon
        laplace_noise = np.random.laplace(loc=0.0, scale=scale, size=lodata.shape)
        lodata_noisy = lodata + laplace_noise
        
        n_clusters = k1
        cluster = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        centers = cluster.fit(lodata_noisy).cluster_centers_
        corepoints.append(centers)
    
    print('Stage 2: Server-side clustering starting')
    
    # Stage 2: Server-side clustering
    serverdata = np.concatenate(corepoints, axis=0)
    label = datapkl['true_label']
    cnum = len(set(true_labels))
    
    print('Stage 3: Final centroid selection starting')
    
    finalcenter, assignment = SNN_optimized(
        cnum, serverdata, candidates_multiplier, energy_multiplier
    )
    
    # Assign all data points to nearest centroids
    distances = cdist(data, finalcenter)
    idx = np.argmin(distances, axis=1) + 1
    
    # Compute clustering metrics
    ari = round(adjusted_rand_score(label, idx), 4)
    nmi = round(normalized_mutual_info_score(label, idx), 4)
    
    return ari, nmi, [n_clusters, cnum]


def run_single_experiment(args):
    """Single experiment runner for multiprocessing"""
    data_path, use_gpu, k1, candidates_multiplier, energy_multiplier, epsilon = args
    return nnfc_optimized(data_path, use_gpu, k1, candidates_multiplier, energy_multiplier, epsilon)


def run_experiments_parallel(data_path, n_runs=1000, n_processes=None, use_gpu=True, 
                           k1=None, dataset=None, candidates_multiplier=None, 
                           energy_multiplier=None, epsilon=None):
    """Run experiments in parallel with batching for large datasets"""
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 8)
    
    print(f"Running {n_runs} experiments on {n_processes} cores...")
    
    results = []
    start_time = time.time()
    
    # Define batch size based on dataset complexity
    batch_size = 5 if dataset in ['celltypes1', 'covtype1', 'postures1', 'mnist2d1'] else n_runs
    batches = (n_runs + batch_size - 1) // batch_size
    
    run_counter = 0
    for batch_idx in range(batches):
        current_batch_size = min(batch_size, n_runs - run_counter)
        args_list = [(data_path, use_gpu, k1, candidates_multiplier, energy_multiplier, epsilon) 
                     for _ in range(current_batch_size)]
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            future_to_run = {executor.submit(run_single_experiment, args): run_counter + i 
                           for i, args in enumerate(args_list)}
            
            for future in as_completed(future_to_run):
                run_id = future_to_run[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if len(results) % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"Completed {len(results)}/{n_runs} runs in {elapsed:.1f}s")
                except Exception as e:
                    print(f"Run {run_id} failed: {e}")
                    results.append((0, 0, [0, 0, 0]))
        
        run_counter += current_batch_size
    
    return results


def evaluate_with_seeds(data_path, use_gpu=True, n_runs=100, n_processes=None, 
                       k1=None, dataset=None, seed=None, candidates_multiplier=None, 
                       energy_multiplier=None, epsilon=None):
    """Evaluate performance across multiple seeds"""
    SEEDS = list(range(seed))
    seed_results = {'ari_max': [], 'nmi_max': []}
    
    for seed in SEEDS:
        print(f"\n--- Evaluating with seed={seed} ---")
        results = run_experiments_parallel(
            data_path, n_runs, n_processes, use_gpu, k1, dataset, 
            candidates_multiplier, energy_multiplier, epsilon
        )
        
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
    print("SEED EVALUATION RESULTS")
    print("="*50)
    print(f"ARI (max): {ari_mean:.4f} ± {ari_std:.4f}")
    print(f"NMI (max): {nmi_mean:.4f} ± {nmi_std:.4f}")
    
    return seed_results


def run_experiment(datanames, seeds, n_runs, use_gpu, save_path, config_file):
    """Main experiment runner"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
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
        
        # Load dataset-specific parameters
        k1 = config[dataset]['k1']
        candidates_multiplier = config[dataset]['candidates_multiplier']
        energy_multiplier = config[dataset]['energy_multiplier']
        epsilon = config['epsilon']
        
        # Run evaluation with multiple seeds
        seed_results = evaluate_with_seeds(
            data_path, use_gpu, n_runs=n_runs, n_processes=n_processes,
            k1=k1, dataset=dataset, seed=seeds, energy_multiplier=energy_multiplier,
            candidates_multiplier=candidates_multiplier, epsilon=epsilon
        )
        
        # Save results
        with open(save_path, 'a') as f:
            f.write(f"{data_path}\n")
            f.write(f"ARI (max) across seeds: {seed_results['ari_max']}\n")
            f.write(f"NMI (max) across seeds: {seed_results['nmi_max']}\n")
            f.write(f"Mean ± Std - ARI: {np.mean(seed_results['ari_max']):.4f} ± {np.std(seed_results['ari_max']):.4f}\n")
            f.write(f"Mean ± Std - NMI: {np.mean(seed_results['nmi_max']):.4f} ± {np.std(seed_results['nmi_max']):.4f}\n\n")