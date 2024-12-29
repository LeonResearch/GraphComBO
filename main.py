from search.run import run_search
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import utils.config_utils as config_utils
import yaml
from itertools import product
from plot_results import plot
from create_path import create_path

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--starting', type=str, default=None)
    parser.add_argument('--problem', type=str, default=None)
    parser.add_argument('--exploitation', type=bool, default=False)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--Q', type=int, default=None)
    parser.add_argument('--failtol', type=int, default=None)
    parser.add_argument('--plot_result', type=bool, default=False)
    parser.add_argument('--ablation', type=bool, default=False)
    parser.add_argument('--noise', type=float, default=None)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)
    
    # load parameters from the defined yaml config file
    if args.problem is not None:
        problem_path = f'./configurations/{args.problem}.yaml'
    else:
        problem_path = f'./configurations/testing.yaml'
    config = config_utils.setup(problem_path)

    # set save path to the specified save_dir in config
    save_dir = config["save_dir"]
    if args.ablation:
        save_dir = save_dir + 'Ablation/'
        config['n_exp'] = 10
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Load some config parameters
    labels = config["label"] # defines the kernel methods
    problem_name=config["problem_name"] # the underlying problem
    problem_kwargs = config["problem_settings"] # settings of the underlying problem
    bo_kwargs = config["bo_settings"] # settings of Bayesian optimisation
    n_exp = getattr(config, "n_exp", 10) # number of repeated experiments
    plot_result = getattr(config, "plot_result", True)
    animate = getattr(config, "animate", False)
    noisy = getattr(problem_kwargs, "noisy", False)
    all_data_over_labels = {l: [] for l in labels}
    
    # ======================== Experimental Settings with args =========================
    # Update the configs with input args if they are specified
    if args.label == 'baselines':
        labels = ["random", "random_walk", "bfs", "dfs", "local_search", "klocal_search"]
    else:
        labels = [args.label] if args.label is not None else labels
    seed=args.start_seed
    problem_kwargs["k"] = args.k if args.k is not None else getattr(problem_kwargs, "k", 2)
    bo_kwargs["start_location"] = args.starting if args.starting is not None else getattr(bo_kwargs,"start_location","random")

    # Settings for ablation studies
    bo_kwargs["Q"] = args.Q if args.Q is not None else bo_kwargs["Q"]
    bo_kwargs["tr_settings"]["fail_tol"] = args.failtol if args.failtol is not None else bo_kwargs["tr_settings"]["fail_tol"]
    # Settings to use a faster method subgraph construction for scale-free networks CS and ba
    if problem_kwargs["graph_type"] in ['CS', 'ba', 'contact_network_day1'] and problem_kwargs["k"] >= 16: 
        bo_kwargs["large_Q"] = True 
    # Settings to set a prior for X by multiplying a weight on the acquisition 
    if args.problem in ["BA", "WS", "SBM", "SIR", "Patient_Zero", "Coauthor_IC"] and problem_kwargs["k"] >= 16:
        use_prior = False
        #bo_kwargs["tr_settings"]["shrink_tol"] = 2 
        bo_kwargs["n_init"] = 10
        if args.problem == 'Coauthor_IC' and problem_kwargs["k"] == 32:
            bo_kwargs["n_init"] = 30
        warmup = "random_walk"
    else:
        use_prior = False
        warmup = bo_kwargs.get("init_method", "random") 
    if noisy:
        problem_kwargs["noise"] = args.noise if (args.noise is not None) else problem_kwargs["noise"]
    
    # ======================== Run ==========================
    save_path = create_path(save_dir, problem_name, problem_kwargs, bo_kwargs)
    all_data = []
    for i in range(n_exp):
        for label_idx, label in enumerate(labels):
            try:
                run_search(
                    label=label,
                    seed=seed + i,
                    problem_name=problem_name,
                    save_path=save_path,
                    start_location_method=getattr(bo_kwargs, "start_location", "random"),
                    restart_location_method=getattr(bo_kwargs,"restart_location","queried_best"),
                    batch_size=getattr(bo_kwargs, "batch_size", 1),
                    n_initial_points=getattr(bo_kwargs, "n_init", 10),
                    iterations=getattr(bo_kwargs, "max_iters", 50),
                    max_radius=getattr(bo_kwargs, "max_radius", 10),
                    Q=getattr(bo_kwargs, "Q", 100),
                    fast_computation=getattr(bo_kwargs, "large_Q", False),
                    exploitation=getattr(bo_kwargs,"exploitation",False),
                    k=problem_kwargs["k"],
                    use_prior=use_prior,
                    noisy=noisy,
                    warmup=warmup,
                    animation=animate,
                    dtype=torch.float32,
                    trust_region_kwargs=getattr(bo_kwargs, "tr_settings", None),
                    problem_kwargs=problem_kwargs,
                )
            except:
                pass
    if args.plot_result:
        plot(save_path, xlim=bo_kwargs["max_iters"])
