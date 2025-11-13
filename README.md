# Topological Federated Clustering via Gravitational Potential Fields (GFC)

🎉 **ACCEPTED AT AAAI 2026 MAIN TECHNICAL TRACK** 🎉

**Topological Federated Clustering via Gravitational Potential Fields under Local Differential Privacy**

📄 [**Paper**](https://arxiv.org/abs/placeholder) | 🌐 [**Project Page**](https://yunbo-max.github.io/GFC/) 

A collection of professional federated clustering algorithms implementing Gravitational Federated Clustering (GFC) and related variants using gravitational potential fields for topological clustering under local differential privacy.

## Code Improvements Made

### ✅ **COMPLETED CLEANUP TASKS**

#### **All Python Files Professionalized:**
- **`main.py`** - Clean experiment runner with proper error handling
- **`models/cfc.py`** - Streamlined CFC implementation 
- **`models/final.py`** - Advanced clustering with proper imports
- **`models/final_ori.py`** - Original variant with tree-based region tracking
- **`models/final_high.py`** - High-dimensional clustering specialization  
- **`models/final_2d.py`** - 2D optimized clustering with visualization
- **`models/nnfc.py`** - Neural Network Federated Clustering
- **`models/kfed.py`** - K-means Federated Clustering with sparse support
- **`models/mufc.py`** - Multi-User Federated Clustering  
- **`models/utils.py`** - Clean utility functions for data processing
- **`models/model.py`** - Professional K-means implementation

#### **Key Improvements:**
- ✅ **Removed excessive comments** - Cut from verbose AI-generated hints to 3-5 essential technical comments per file
- ✅ **Eliminated redundant imports** - Removed duplicate and unused imports, organized logically  
- ✅ **Professional module documentation** - Added clean docstrings describing functionality
- ✅ **Consistent code formatting** - Professional style suitable for open source
- ✅ **Optional dependency handling** - Graceful handling of GPU libraries (cupy, cuml), visualization (anytree, umap)
- ✅ **Function signature cleanup** - Simplified overly verbose docstrings to concise descriptions

### 📁 **File Structure**

```
├── main.py                    # Unified experiment runner
├── configs/                   # JSON configuration files  
├── models/
│   ├── cfc.py                # Core CFC implementation
│   ├── final.py              # Advanced multi-dimensional clustering
│   ├── final_ori.py          # Tree-based region tracking variant
│   ├── final_high.py         # High-dimensional data specialization
│   ├── final_2d.py           # 2D visualization optimized
│   ├── nnfc.py               # Neural Network Federated Clustering
│   ├── kfed.py               # K-means with sparse matrix support
│   ├── mufc.py               # Multi-User Federated Clustering
│   ├── model.py              # Custom K-means implementation
│   └── utils.py              # Data loading and processing utilities
└── dataset/                  # Data files (not tracked)
```

### 🚀 **Core Features**

- **Differential Privacy** - Laplace noise for privacy preservation
- **Synthetic Centroids** - Energy-based centroid generation using electrical potential analogies
- **Multi-processing** - Parallel execution for large-scale experiments  
- **Optional GPU Acceleration** - CUDA support when available (cupy/cuml)
- **Flexible Configuration** - JSON-based parameter management
- **Multiple Variants** - Specialized implementations for different data types

### 📊 **Usage Examples**

```bash
# Standard CFC clustering
python main.py --config cfc --ep 0.1

# High-dimensional data
python main.py --config final --ep 0.01  

# Neural network approach
python main.py --config nnfc --ep 1.0

# K-means federated
python main.py --config kfed --ep 0.05
```

### 🔧 **Configuration Parameters**

Each algorithm supports customizable parameters via JSON configs:
- **`k1`, `k2`** - Cluster counts for multi-stage clustering
- **`candidates_multiplier`** - Synthetic candidate density 
- **`energy_multiplier`** - Energy function scaling
- **`epsilon`** - Differential privacy budget
- **`n_runs`** - Number of experimental runs
- **`seeds`** - Random seed range for reproducibility

### 📈 **Evaluation & Metrics**

- **ARI (Adjusted Rand Index)** - Clustering accuracy against ground truth
- **NMI (Normalized Mutual Information)** - Information-theoretic clustering quality
- **Multi-seed Statistics** - Mean ± standard deviation across random seeds
- **Parallel Execution** - Efficient evaluation across multiple cores

### 🎯 **Professional Standards Achieved**

- **Clean Architecture** - Modular design with clear separation of concerns
- **Error Handling** - Graceful failure with informative messages  
- **Documentation** - Essential comments only, focused on technical insights
- **Dependency Management** - Optional imports with fallback handling
- **Code Style** - Consistent formatting and naming conventions
- **Open Source Ready** - Professional appearance suitable for publication

The codebase is now **production-ready** with clean, maintainable code that focuses on algorithmic innovation rather than verbose explanations.