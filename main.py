# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2025-07-04 00:11:55
# @Last Modified by:   Yunbo
# @Last Modified time: 2025-07-04 13:48:48
import argparse
import torch
import importlib
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner with config file")
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    parser.add_argument('--ep', type=float, default=1000)
    args = parser.parse_args()

    # Construct full config path
    config_file = f"configs/{args.config}_config_ep{args.ep}.json"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found.")
    
    with open(config_file, 'r') as f:
        config = json.load(f)

    model_name = config['model']
    datanames = config['datanames']
    seeds = config.get('seed', 100)
    n_runs = config.get('n_runs', 1)
    save_path = config.get('save_path', 'results/default_results.txt')

    use_gpu = torch.cuda.is_available()

    print('seeds ',seeds)

    # try:
    model_module = importlib.import_module(f"models.{model_name}")
    model_module.run_experiment(
        datanames=datanames,
        seeds=seeds,
        n_runs=n_runs,
        use_gpu=use_gpu,
        save_path=save_path,
        config_file=config_file
    )
    # except ModuleNotFoundError:
    #     print(f"Model module '{model_name}' not found. Please ensure it's in models/.")
    # except AttributeError:
    #     print(f"'run_experiment' not found in {model_name}. Please define it.")

if __name__ == '__main__':
    main()
