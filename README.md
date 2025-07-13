# 🌀 CFC: Federated Clustering Framework

![License](https://img.shields.io/badge/license-MIT-green)


## 🛠 Installation

### Prerequisites
- Conda (recommended)
- Python 3.10+

### Setup Environment
```bash
conda create -n cfc python=3.10 -y
conda activate cfc
pip install -r requirements.txt
```

###  Dataset Preparation
1. Create dataset directory:
   ```bash
   mkdir -p datasets
   ```
  
2. Download datasets from these sources:

- NNFC Datasets: https://github.com/mlyizhang/nnfc

- MUFC Datasets: https://github.com/thupchnsky/mufc

3. Place the downloaded datasets in the datasets/ folder

### Setup Environment
configs/
└── cfc_config_ep1000.json

```bash
{
  "model": "cfc",
  "datanames": ["mnist2d"],
  "seed": 1,
  "n_runs": 1,
  "save_path": "results/output.txt",
  "mnist2d": {
    "k1": 200,
    "energy_multiplier": 13,
    "energy_multiplier": 13,
      "energy_threshold":90
  }
```

### Running experiments
```bash
python main.py --config cfc --ep 1000
python main.py --config cfc --ep 1
python main.py --config cfc+ --ep 0.1
python main.py --config nnfc --ep 0.01
.........
```