# CFC: Federated Clustering Framework
==================================

╔════════════════════════════════════════════════════╗
║  ██████╗ ██████╗ ██████╗   FEDERATED CLUSTERING   ║
║ ██╔════╝██╔═══██╗██╔══██╗      FRAMEWORK         ║
║ ██║     ██║   ██║██████╔╝                        ║
║ ██║     ██║   ██║██╔═══╝                         ║
║ ╚██████╗╚██████╔╝██║                              ║
║  ╚═════╝ ╚═════╝ ╚═╝                              ║
╚════════════════════════════════════════════════════╝

✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦

🚀 SETUP INSTRUCTIONS
───────────────────────────────────────────────────

1. Create Conda Environment:
   └─▶ conda create -n cfc_env python=3.8 -y
   └─▶ conda activate cfc_env

2. Install Dependencies:
   └─▶ pip install -r requirements.txt

⚙️ CONFIGURATION
───────────────────────────────────────────────────
Config file location:
   └─▶ /Users/yunbo/Documents/GitHub/CFC_Federated_Clustering/
       CFC_Federated_Clustering/configs/cfc_config_ep1000.json

Sample Config:
▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
{
  "model": "cfc",
  "datanames": ["mnist2d"],
  "seed": 1,
  "n_runs": 1,
  "epsilon": 1,
  "save_path": "results/cfc_epsilon_1.txt",
  "mnist2d": {
    "k1": 200,
    "candidates_multiplier": 2,
    "energy_multiplier": 13
  }
}
▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

🏃 RUNNING THE FRAMEWORK
───────────────────────────────────────────────────
Basic Command:
   ┌─▶ python main.py --config cfc --ep 1000
   └─── (Uses default 'cfc' config with 1000 epochs)

Example with kfed config:
   ┌─▶ python main.py --config kfed --ep 1000
   └─── (Runs with 'kfed' configuration)

Command Arguments:
┌──────────────────┬──────────┬───────────┬─────────────────────┐
│    Argument      │   Type   │  Default  │     Description     │
├──────────────────┼──────────┼───────────┼─────────────────────┤
│ --config         │ string   │ "cfc"     │ Configuration name  │
│ --ep             │ float    │ 1000      │ Number of epochs    │
└──────────────────┴──────────┴───────────┴─────────────────────┘

📂 PROJECT STRUCTURE
───────────────────────────────────────────────────
CFC_Federated_Clustering/
├── 📁 configs/       # JSON configuration files
├── 📁 results/       # Output directory
├── 📄 main.py        # Main implementation
└── 📄 requirements.txt # Dependencies

✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦