"""
K-means Federated Clustering (KFed) Implementation
Federated clustering using k-means with sparse matrix support.
"""
import numpy as np
import scipy
import scipy.sparse as sps
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances as sparse_cdist
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Optional GPU support
try:
    import cupy as cp
    import cuml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def distance_to_set(A, S, sparse=False):
    """Compute minimum distance from each point in A to the set S"""
    n, d = A.shape
    assert S.ndim == 2
    assert S.shape[1] == d, S.shape[1]
    assert A.shape[1] == d
    assert A.ndim == 2
    # Pair wise distances
    if sparse is False:
        pd = scipy.spatial.distance.cdist(A, S, metric='euclidean')
    else:
        pd = sparse_cdist(A, S)
    assert np.allclose(pd.shape, [A.shape[0], len(S)])
    dx = np.min(pd, axis=1)
    assert len(dx) == A.shape[0]
    assert dx.ndim == 1
    return dx


def get_clustering(A, centers, sparse=False):
    '''
    Returns a list of integers of length len(A). Each integer is an index which
    tells us the cluster A[i] belongs to. A[i] is assigned to the closest
    center.
    '''
    # Pair wise distances
    if sparse is False:
        pd = scipy.spatial.distance.cdist(A, centers, metric='euclidean')
    else:
        pd = sparse_cdist(A, centers)
    assert np.allclose(pd.shape, [A.shape[0], len(centers)])
    indices = np.argmin(pd, axis=1)
    assert len(indices) == A.shape[0]
    return np.array(indices)


def kmeans_cost(A, centers, sparse=False, remean=False):
    '''
    Computes the k means cost of rows of $A$ when assigned to the nearest
    centers in `centers`.

    remean: If remean is set to True, then the kmeans cost is computed with
    respect to the actual means of the clusters and not necessarily the centers
    provided in centers argument (which might not be actual mean of the
    clustering assignment).
    '''
    clustering = get_clustering(A, centers, sparse=sparse)
    cost = 0
    if remean is True:
        # We recompute mean based on assignment.
        centers2 = []
        for clusterid in np.unique(clustering):
            points = A[clustering == clusterid]
            centers2.append(np.mean(points, axis=0))
        centers = np.array(centers2)
    for clusterid in np.unique(clustering):
        points = A[clustering == clusterid]
        dist = distance_to_set(points, centers, sparse=sparse)
        cost += np.mean(dist ** 2)
    return cost


def kmeans_pp(A, k, weighted=True, sparse=False, verbose=False):
    '''
    Returns $k$ initial centers based on the k-means++ initialization scheme.
    With weighted set to True, we have the standard algorithm. When weighted is
    set to False, instead of picking points based on the D^2 distribution, we
    pick the farthest point from the set (careful deterministic version --
    affected by outlier points). Note that this is not deterministic.

    A: nxd data matrix (sparse or dense). 
    k: is the number of clusters.

    Returns a (k x d) dense matrix.

    K-means ++
    ----------
     1. Choose one center uniformly at random among the data points.
     2. For each data point x, compute D(x), the distance between x and
        the nearest center that has already been chosen.
     3. Choose one new data point at random as a new center, using a
        weighted probability distribution where a point x is chosen with
        probability proportional to D(x)2.
     4. Repeat Steps 2 and 3 until k centers have been chosen.
    '''
    n, d = A.shape
    if n <= k:
        if sparse:
            A = A.toarray()
        return np.array(A)  # Fixed typo: np.aray -> np.array
    index = np.random.choice(n)
    if sparse is True:
        B = np.squeeze(A[index].toarray())
        assert len(B) == d
        inits = [B]
    else:
        inits = [A[index]]
    indices = [index]
    t = [x for x in range(A.shape[0])]
    distance_matrix = distance_to_set(A, np.array(inits), sparse=sparse)
    distance_matrix = np.expand_dims(distance_matrix, axis=1)
    while len(inits) < k:
        if verbose:
            print('\rCenter: %3d/%4d' % (len(inits) + 1, k), end='')
        # Instead of using distance to set we can compute this incrementally.
        dx = np.min(distance_matrix, axis=1)
        assert dx.ndim == 1
        assert len(dx) == n
        dx = dx**2/np.sum(dx**2)
        if weighted:
            choice = np.random.choice(t, 1, p=dx)[0]
        else:
            choice = np.argmax(dx)
        if choice in indices:
            continue
        if sparse:
            B = np.squeeze(A[choice].toarray())
            assert len(B) == d
        else:
            B = A[choice]
        inits.append(B)
        indices.append(choice)
        last_center = np.expand_dims(B, axis=0)
        assert last_center.ndim == 2
        assert last_center.shape[0] == 1
        assert last_center.shape[1] == d
        dx = distance_to_set(A, last_center, sparse=sparse)
        assert dx.ndim == 1
        assert len(dx) == n
        dx = np.expand_dims(dx, axis=1)
        a = [distance_matrix, dx]
        distance_matrix = np.concatenate(a, axis=1)
    if verbose:
        print()
    return np.array(inits)


def awasthisheffet(A, k, useSKLearn=True, sparse=False):
    '''
    The implementation here uses kmeans++ (i.e. probabilistic) to get initial centers 
    (\nu in the paper) instead of using a 10-approx algorithm.

    1. Project onto $k$ dimensional space.
    2. Use $k$-means++ to initialize.
    3. Use 1:3 distance split to improve initialization.
    4. Run Lloyd steps and return final solution.

    Returns a sklearn.cluster.Kmeans object with the clustering information and
    the list $S_r$.
    '''
    assert A.ndim == 2
    n = A.shape[0]
    d = A.shape[1]
    # If we don't have $k$ points then return the matrix as its the best $k$
    # partition trivially.
    if n <= k:
        if sparse:
            A = np.array(A.toarray())
        return A, None
    
    # Determine the projection dimension - use min(k, d) to avoid issues
    proj_dim = min(k, d)
    
    # This works with sparse and dense matrices. Returns dense always.
    # Randomized though so average.
    U, Sigma, V = randomized_svd(A, n_components=proj_dim, random_state=None)
    # Columns of $V$ are eigen vectors
    V = V.T[:, :proj_dim]
    # Sparse and dense compatible. A_hat is always dense.
    A_hat = A.dot(V)
    inits = kmeans_pp(A_hat, k, sparse=False)
    # Run STEP 2, modified Lloyd. We have vectorized it for speed up.
    if sparse is False:
        pd = scipy.spatial.distance.cdist(inits, A_hat)
    else:
        pd = sparse_cdist(inits, A_hat)
    Sr_list = []
    for r in range(k):
        th = 3 * pd[r, :]
        remaining_dist = pd[np.arange(k) != r]
        assert np.allclose(remaining_dist.shape, [k- 1, n])
        indicator = (remaining_dist - th) < 0
        indicator = np.sum(indicator.astype(int), axis=0)
        assert len(indicator) == n
        # places where indicator is 0 is our set
        Sr = [i for i in range(len(indicator)) if indicator[i] == 0]
        assert len(Sr) >= 0
        Sr_list.append(Sr)
    
    # Handle empty clusters by assigning at least one point
    for i, Sr in enumerate(Sr_list):
        if len(Sr) == 0:
            # Find the closest point to the init center for this cluster
            distances_to_init = np.linalg.norm(A_hat - inits[i], axis=1)
            closest_point_idx = np.argmin(distances_to_init)
            Sr_list[i] = [closest_point_idx]
    
    # We don't mind lloyd_init being dense. Its only k x proj_dim.
    lloyd_init = np.array([np.mean(A_hat[Sr], axis=0) for Sr in Sr_list])
    assert np.allclose(lloyd_init.shape, [k, proj_dim])
    # Project back to d dimensional space
    lloyd_init = np.matmul(lloyd_init, V.T)
    assert np.allclose(lloyd_init.shape, [k, d])
    # Run Lloyd's method
    if useSKLearn:
        # Works with sparse matrices as well.
        kmeans = KMeans(n_clusters=k, init=lloyd_init, n_init=1)
        kmeans.fit(A)
        ret = (kmeans.cluster_centers_, kmeans)
    else:
        raise NotImplementedError()
    # We use the GPU version from torch:with
    return ret


def kfed(x_dev, dev_k, k, useSKLearn=True, sparse=False):
    '''
    The full decentralized algorithm.

    Warning: Synchronous version, no parallelization across devices. Since the
    sklearn k means routine is itself parallel. 

    x_dev: [Number of devices, data length, data dimension]
    dev_k: Device k (int). The value $k'$ in the paper. Number of clusters
        per device. We use constant for all devices.

    https://further-reading.net/2017/01/quick-tutorial-python-multiprocessing/

    Returns: Local estimators (local centers), central-centers
    '''
    def cleaup_max(local_estimators, k, dev_k, useSKLearn=True, sparse=False):
        '''
        Central cleanup phase based on the max-from-set rule.
        
        Switch to either percentile rule or probabilistic (kmeans++) rule in
        case of outlier points.
        '''
        assert local_estimators.ndim == 2
        # The first dev_k points definitely in different target clusters.
        init_centers = local_estimators[:dev_k, :]
        remaining_data = local_estimators[dev_k:, :]
        # For the remaining initialization, use max rule.
        while len(init_centers) < k:
            distances = distance_to_set(remaining_data, np.array(init_centers),
                                        sparse=sparse)
            candidate_index = np.argmax(distances)
            candidate = remaining_data[candidate_index:candidate_index+1, :]
            # Combine with init_centers
            init_centers = np.append(init_centers, candidate, axis=0)
            # Remove from remaining_data
            remaining_data = np.delete(remaining_data, candidate_index, axis=0)

        assert len(init_centers) == k
        # Perform final clustering.
        if useSKLearn:
            # Works with sparse matrices as well.
            kmeans = KMeans(n_clusters=k, init=init_centers)
            kmeans.fit(local_estimators)
            ret = (kmeans.cluster_centers_, kmeans)
        else:
            raise NotImplementedError("This is not implemented/tested")
        return ret

    num_dev = len(x_dev)
    msg = "Not enough devices "
    msg += "(num_dev=%d, dev_k=%d, k=%d)" % (num_dev, dev_k, k)
    assert dev_k * num_dev >= k, msg
    # Run local $k$-means
    local_clusters = []
    for dev in x_dev:
        cluster_centers, _ = awasthisheffet(dev, dev_k, useSKLearn=useSKLearn,
                                            sparse=sparse)
        local_clusters.append(cluster_centers)
    # This is always dense.
    local_estimates = np.concatenate(local_clusters, axis=0)
    msg = "Not enough estimators. "
    msg += "Estimator matrix size: " + str(local_estimates.shape) + ", while "
    msg += "k = %d" % k
    assert local_estimates.shape[0] > k, msg
    # Local estimators are dense
    centers, kmeansobj = cleaup_max(local_estimates, k, dev_k,
                                    useSKLearn=useSKLearn, sparse=False)
    return local_estimates, centers


def load_dataset(data_path):
    """Load federated dataset from pickle file"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def run_single_experiment(args):
    """Run a single experiment with given parameters"""
    data_path, seed, k1, k2, use_gp,epsilon = args
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    try:
        # Load dataset according to your format
        datapkl = load_dataset(data_path)
        true_labels = np.array(datapkl['true_label'])  # Original labels for full data
        
        # Extract data according to your structure
        # eachlable = datapkl['eachlable']
        # order = datapkl['order']
        data = np.array(datapkl['full_data'])
        print(f"Data shape: {data.shape}")
        
        # Get true labels for evaluation
        label = datapkl['true_label']
        k_true = len(np.unique(label))  # Number of true clusters
        
        # Prepare federated data with noise (as in your code)
        scale = 1 / epsilon
        x_fed = []  # List to store client data
        
        print('Preparing federated data with differential privacy noise')
        for i_client in range(10):  # 10 clients as in your code
            print(f'Processing client {i_client}')
            lodata = datapkl["client_" + str(i_client)]
            
            # Add Laplace noise for differential privacy
            laplace_noise = np.random.laplace(loc=0.0, scale=scale, size=lodata.shape)
            lodata_noisy = lodata + laplace_noise
            
            x_fed.append(lodata_noisy)
        

        serverdata = np.concatenate(x_fed, axis=0)
        
        # For the centroids' true labels, we need to assign them based on their closest true data points
        centroid_true_labels = []
        for centroid in serverdata:
            # Find the closest original data point to this centroid
            closest_idx = np.argmin(np.linalg.norm(data - centroid, axis=1))
            centroid_true_labels.append(true_labels[closest_idx])
        centroid_true_labels = np.array(centroid_true_labels)
            
        print('Running federated k-means (kfed algorithm)')
        # Run federated k-means with your parameters
        # k1 is dev_k (clusters per device), k_true is final number of clusters
        local_estimates, final_centers = kfed(x_fed, k1, k_true, useSKLearn=True, sparse=False)

        # # Visualization with true labels
        # plot_clusters(serverdata, final_centers, energy=None, true_labels=centroid_true_labels)

        
            
        print(f'Final centers shape: {final_centers.shape}')
        print(f'Full data shape: {data.shape}')
        
        # Calculate distances and assignments (similar to your SNN_optimized approach)
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Get assignments for all data points
        distances = euclidean_distances(data, final_centers)
        print('Calculating final assignments')
        
        # Get cluster assignments (0-based indexing first)
        idx_0based = np.argmin(distances, axis=1)  # Find closest centroid
        idx = idx_0based + 1  # Convert to 1-based indexing as in your code
        
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
    

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

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







def run_experiments_parallel(data_path, n_runs, n_processes, use_gpu, k1, k2,epsilon,seed):
    """Run experiments in parallel across different random seeds"""
    # Generate random seeds for experiments

    
    # Prepare arguments for parallel processing
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

        # Start timer
        start_time = time.time()
        
        results = run_experiments_parallel(data_path, n_runs, n_processes, use_gpu, k1, k2, epsilon, seed)

        # Calculate and print elapsed time
        elapsed_time = time.time() - start_time
        print(f"Completed in: {elapsed_time:.2f} seconds")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
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

    # dataname = ['postures']
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
        print(f"Using k1={k1}, k2={k2} for dataset {dataset}")
        
        # Evaluate with multiple seeds
        seed_results = evaluate_with_seeds(data_path, use_gpu, n_runs=n_runs, n_processes=n_processes, k1=k1, k2=k2,seed=seeds,epsilon=epsilon)
        
        # Save seed results
        with open(save_path, 'a') as f:
            f.write(f"{data_path}\n")
            f.write(f"ARI (max) across seeds: {seed_results['ari_max']}\n")
            f.write(f"NMI (max) across seeds: {seed_results['nmi_max']}\n")
            f.write(f"Mean ± Std - ARI: {np.mean(seed_results['ari_max']):.4f} ± {np.std(seed_results['ari_max']):.4f}\n")
            f.write(f"Mean ± Std - NMI: {np.mean(seed_results['nmi_max']):.4f} ± {np.std(seed_results['nmi_max']):.4f}\n\n")

