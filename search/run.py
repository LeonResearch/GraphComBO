import numpy as np
import networkx as nx
import torch
import random
import os
import math
import matplotlib.pyplot as plt
from time import time
from decimal import Decimal
from typing import Optional, Dict, Any
from problems.underlying_problem import get_synthetic_problem
from search.baselines import Baseline
from search.graphcombo import GraphComBO
#from search.recursive_sampling import ComboSubgraph_Constructor
from search.self_combograph import ComboSubgraph_Constructor
from search.trust_region import update_state, restart
from search.utils import (
    prune_baseline,
    ComboNeighbors_Generator,
    generate_start_location,
    eigendecompose_laplacian,
    filter_invalid,
    Index_Mapper,
    get_X_prior,
    prior_decay,
)


def run_search(
    label: str,
    seed: int,
    problem_name: str,
    save_path: str,
    iterations: int = 100,  # Number of queries in the search
    batch_size: int = 1,
    n_initial_points: int = 1,
    start_location_method: str = "random",
    restart_location_method: str = "same_as_start",
    acqf_optimizer: str = "enumerate",
    max_radius: int = 10,  # max radius in the ComboSubgraph
    Q: int = 1000,  # Size of the ComboSubgraph
    fast_computation: bool = False,  # if True we use a faster way to compute the ComboSubgraph
    exploitation: bool = False,
    k: int = 2,  # Number of combinations
    use_prior: bool = False,
    warmup: str = 'random',
    noisy: bool = False,
    X_prior = None,
    acqf_kwargs: Optional[dict] = None,
    acqf_optim_kwargs: Optional[dict] = None,
    model_optim_kwargs: Optional[Dict[str, Any]] = None,
    trust_region_kwargs: Optional[Dict[str, Any]] = None,
    problem_kwargs: Optional[Dict[str, Any]] = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    save_frequency: int = 1,
    animation: bool = False,
    animation_interval: int = 20,
    order=None,
):
    graph_kernels = ["polynomial","polynomial_suminverse",
                     "diffusion","diffusion_ard",]
    baselines = ["random", "random_walk", "dfs", "bfs", "local_search", "klocal_search"]
    trust_region_kwargs = trust_region_kwargs or {}
    problem_kwargs = problem_kwargs or {}
    save_path = os.path.join(save_path, label)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    tkwargs = {"dtype": dtype, "device": device}
    acqf_optim_kwargs = acqf_optim_kwargs or {}
    acqf_kwargs = acqf_kwargs or {}
    acqf_kwargs["type"] = "ucb" if exploitation else "ei"
    model_optim_kwargs = model_optim_kwargs or {}

    if use_prior:
        from search.recursive_sampling import ComboSubgraph_Constructor
    else:
        from search.self_combograph import ComboSubgraph_Constructor
    
    # -------------------- Initialise the Problem --------------------
    Problem = get_synthetic_problem(problem_name, seed=seed, problem_kwargs=problem_kwargs)
    print(
        f"Using {label} model with Q={Q} radius={max_radius} failtol={trust_region_kwargs['fail_tol']} "
        f"start={start_location_method} restart={restart_location_method} exploitation={exploitation}"
    )
    ground_truth = None if Problem.ground_truth is None else Problem.ground_truth.cpu()
    use_trust_region = label in graph_kernels
    n_restart = 0
    n_contextsubgraph = 0
    if start_location_method in ["ei", "betweenness", "degree", "pagerank",]:  
        start_location = generate_start_location(
            Problem.underlying_graph, k, start_location_method
        )
    else: # Use a random start location
        start_location = None

    # Generate initial starting location
    candidates, trust_region_state = restart(
        base_graph=Problem.underlying_graph,
        n_init=n_initial_points,
        seed=seed,
        k=k,
        context_subgraph_size=Q,  # Initial subgraph size
        X_avoid=None,
        anchor=start_location,
        initialization=True,
        warmup=warmup,
        use_trust_region=use_trust_region,
        options=trust_region_kwargs,
    )
    # X_queried/Y_queried is the set of queried nodes/values.
    # X_queried is the global indices (i.e. on the original graph).
    X_queried = candidates.to(**tkwargs)  
    Y_queried = Problem(X_queried).to(**tkwargs)  
    # X_train, Y_train are the training sets, i.e. nodes inside subgraph at each iter.
    # They will be updated at each iteration by "prune_baseline".
    X_train = X_queried.clone()  
    Y_train = Y_queried.clone()  
    # Note X_train/Y_train are useless for most baselines except (k)local search
    # Set some counters to keep track of things.
    n_restart += 1
    existing_iterations = 0
    wall_time = []
    threshold_obj = Y_queried.max().view(-1)[0].cpu().numpy()
    queried_best_loc = X_queried[Y_queried.argmax().view(-1)[0]].cpu()
    start_combonode = (
        tuple(queried_best_loc.to(int).tolist())
        if not exploitation
        else tuple(candidate_start.squeeze().tolist())
    )
    center = torch.tensor(start_combonode)
    # record the starting combo_node Y so no need to query it later
    start_combonode_Y = (
        Y_queried[Y_queried.argmax()].reshape(1, -1)
        if not exploitation
        else Y_queried[0].reshape(1, -1)
    )
    combosubgraph_center_degree = len(
        ComboNeighbors_Generator(
            Problem.underlying_graph, torch.tensor(start_combonode)
        )
    )  # track the degree
    # Q = max(int(combosubgraph_center_degree/4), Q) if exploitation else Q
    if acqf_optimizer is None:
        acqf_optimizer = "enumerate" if Problem.problem_size <= 1000 else "local_search"

    # --------------------- Initialise the Models --------------------
    # initialise BO models
    if label in graph_kernels:
        ComboSubgraph, n_hop = ComboSubgraph_Constructor(
            Problem.underlying_graph,
            torch.tensor(start_combonode).numpy().reshape(1, -1),
            nx.Graph(),
            Q=Q,
            l_max=max_radius,
            fast_computation=fast_computation,
        )
        ComboGraph = nx.Graph()
        # Keep tracking the total explored combo graph
        ComboGraph.add_edges_from(list(ComboSubgraph.edges))  
        # Only include queried nodes that are inside ComboSubgraph for training
        idx_keep = torch.tensor(list(ComboSubgraph.nodes))
        X_train, Y_train = prune_baseline(X_queried, Y_queried, idx_keep)
        # Functions to map between tuple node-label and int node-label. This is for kernels in graph GP,
        # which will regard int node labels as "features", and use these features as "indices" to choose
        # which eigenvector elements to use, i.e., when computing covariance between node i and node j, 
        # we need to use engenvector_matrix[i] and eigenvector_matrix[j] in the kernel functions.
        # mapping and remapping dictionaries
        tuple_to_int, int_to_tuple = Index_Mapper(ComboSubgraph)  
        def tuple_to_int_mapper(x):  # map node labels from tuples to integers
            ret = [tuple_to_int[tuple(i.int().tolist())] for i in x]
            return torch.tensor(ret).to(x).reshape(-1, 1)
        def int_to_tuple_mapper(x):
            # map node labels from integers back to tuple
            return torch.tensor([int_to_tuple[int(i)] for i in x]).to(x)

        if use_prior: # Generate some prior preference for acquisition
            X_prior = get_X_prior(ComboSubgraph, start_combonode)
        # Set up the configurations for surrogate models with different kernels
        model_configurations = {"covar_type": label, "order": None, "ard": True}
        if label in ["diffusion_ard"]:  
            # both diffusion_ard and diffusion share the same kernel
            # the differences are "ard" and "order"
            model_configurations["covar_type"] = "diffusion"  
            # change to order size context graph
            model_configurations["order"] = len(ComboSubgraph.nodes)  
        elif label == "diffusion":
            model_configurations["order"] = len(ComboSubgraph.nodes)  
            model_configurations["ard"] = False  # Diffusion kernel without ARD
        elif label in ["polynomial_suminverse", "polynomial"]:
            model_configurations["order"] = n_hop  ## Change to order size context graph

            torch.tensor(start_combonode).numpy().reshape(1, -1),
    
    # Initialise baselines
    elif label in baselines:
        ComboSubgraph = (
            nx.Graph() if label == "local_search" else None
        )  # local context combo-subgraph is not needed in baselines
        ComboGraph = nx.Graph()
        ComboGraph.add_nodes_from([start_combonode])
        num_explored_combonodes = 0
        if label in ["dfs", "bfs"]:  # Initialise the stacks for DFS/BFS
            list_stacks = []
            neighbors_current = ComboNeighbors_Generator(
                Problem.underlying_graph,
                torch.tensor(start_combonode),
                X_avoid=X_queried,
            )
            neighbors_current_tuple = [tuple(x) for x in neighbors_current.tolist()]
            # make it random for multiple experiments
            random.shuffle(neighbors_current_tuple)  
            list_stacks.append(neighbors_current_tuple)
            # Now we add these new graph structure to ComboGraph for recording purposes
            center_combonode_to_stack = np.repeat(
                np.array(start_combonode).reshape(1, -1),
                len(neighbors_current),
                axis=0,
            )
            comboedges_array_to_add = np.stack(
                (neighbors_current, center_combonode_to_stack), axis=1
            ).astype(int)
            # Convert np.array to list of tuples
            comboedges_to_add = list(tuple(map(tuple,i)) for i in comboedges_array_to_add)  
            # record the explored structure
            ComboGraph.add_edges_from(comboedges_to_add)  

            num_explored_combonodes = len(
                ComboSubgraph_Constructor(
                    Problem.underlying_graph,
                    [start_combonode],
                    nx.Graph(),
                    Q=1e7,
                    l_max=1,
                    fast_computation=fast_computation,
                )[0]
            )
            hop_tracker = 1

    # Set some experiment recorders
    cached_eigenbasis = None
    converge = False
    iter = 1
    sss = time()
    exploration_list = []
    degree_list = []
    distance_list = []
    # -------------------------------- Search Starts -----------------------------------
    while len(X_queried) < iterations:
        start_time = time()
        anchor = (
            queried_best_loc.unsqueeze(0)
            if (restart_location_method == "queried_best" and label in graph_kernels)
            else start_location
        )
        # we will only evaluate 1 location at restart.
        n_initial_points = 1  
        # -------------------------- Select Query Location ----------------------------
        # Use baseline methods
        if label in baselines:
            baseline_methods = Baseline(
                Graph=Problem.underlying_graph, 
                ComboGraph=ComboGraph, 
                k=k,
                X_queried=X_queried,
                X_train=X_train, 
                Y_train=Y_train, 
                seed=seed,
                n_init=n_initial_points,
            ) # Initialise the Baseline class
            if label == "random":
                candidates = baseline_methods.Random_Sample(n_samples=iterations)
            elif label == "random_walk":
                candidates = baseline_methods.kRandom_Walk()
            elif label == "klocal_search":
                candidates, X_train, Y_train = baseline_methods.kLocal_Search()
            elif label in ["bfs", "dfs"]:
                candidates, ComboGraph, list_stacks = baseline_methods.DFS_BFS_Search(label, list_stacks)
            elif label == "local_search":
                candidates, X_train, Y_train, ComboGraph = baseline_methods.Local_Search()
        # Use GraphComBO 
        elif label in graph_kernels:
            model = GraphComBO(
                Graph=Problem.underlying_graph,
                ComboSubgraph=ComboSubgraph,
                kernel=label,
                k=k,
                Q=Q,
                anchor=anchor,
                n_init=n_initial_points,
                seed=seed,
                converge=converge,
                noisy=noisy,
                tuple_to_int_mapper=tuple_to_int_mapper,
                int_to_tuple_mapper=int_to_tuple_mapper,
                kernel_configurations=model_configurations,
                cached_eigenbasis=cached_eigenbasis,
                trust_region_state=trust_region_state,
                trust_region_kwargs=trust_region_kwargs,
                acqf_kwargs=acqf_kwargs,
                acqf_optim_kwargs=acqf_optim_kwargs,
                model_optim_kwargs=model_optim_kwargs,
            )
            X_prior = prior_decay(iterations, iterations/10, iter) * X_prior if use_prior else None
            candidates, X_train, Y_train, ComboSubgraph = model.get_candidates(X_train,Y_train,X_queried,X_prior)
            trust_region_state = model.trust_region_state
            cached_eigenbasis = model.cached_eigenbasis
        else:
            raise NotImplementedError(f"Method {label} is not implemented!")

        # ------------------------- Evaluate Query Location ---------------------------
        if candidates is None:
            converge = True
            n_restart += 1
            continue  # Skip the sections below when all combo-nodes in the current subgraph are queried.
        else:
            converge = False

        new_y = Problem(candidates)  # Query the selected location

        # if X_train.shape[0] == 0 and anchor is not None:
        candidate_in_query = (
            (candidates.squeeze().long() == X_queried.long()).all(dim=-1).sum().item()
            if label in graph_kernels
            else False
        )
        if candidate_in_query:
            pass  # do not append the candidate if it is a previously qeuried anchor point at restart
        else: # append the new queried point to the queried set
            X_queried = torch.cat([X_queried, candidates], dim=0)  
            Y_queried = torch.cat([Y_queried, new_y], dim=0)
        # append the new queried point to the training set
        X_train = torch.cat([X_train, candidates], dim=0)
        Y_train = torch.cat([Y_train, new_y], dim=0)  

        new_best_obj = Y_train.max().squeeze().cpu().numpy()
        queried_best_obj = Y_queried.max().squeeze().cpu().numpy()
        queried_best_loc = X_queried[Y_queried.argmax().cpu()]
        
        # %%%%%%%%%%%%%%%%%% Experiment Recorder %%%%%%%%%%%%%%%%%%%%%
        # Here we record the distance of the current combonode from the starting combonode
        if label in graph_kernels + ["dfs", "bfs", "local_search"]:
            num_explored_combonodes = (
                len(ComboGraph.nodes()) if label != "bfs" else num_explored_combonodes
            )
            try:
                distance_from_start = nx.shortest_path_length(
                    ComboGraph,
                    source=start_combonode,
                    target=tuple(candidates.long().tolist()[0]),
                )
            except:
                distance_from_start = 0
        else:
            distance_from_start = 0

        # Record the exploration size and the current combosubgraph center degree for bfs & dfs
        if label in ["bfs", "dfs"]:
            if distance_from_start > hop_tracker:
                hop_tracker = distance_from_start
                combosubgraph_center_degree = len(ComboNeighbors_Generator(
                    Problem.underlying_graph, candidates.reshape(-1,)
                ))  # track the degree
                if label == "bfs":
                    center_combonodes = np.vstack(
                        [candidates.numpy(), np.array(start_combonode).reshape(1, -1)]
                    )
                    num_explored_combonodes = len(
                        ComboSubgraph_Constructor(
                            Problem.underlying_graph,
                            center_combonodes,
                            nx.Graph(),
                            Q=1e7,
                            l_max=1,
                            fast_computation=fast_computation,
                        )[0]
                    )
        # %%%%%%%%%%%%%%%%%% End of Experiment Recorder %%%%%%%%%%%%%%%%%%%%%

        wall_time.append(time() - start_time)
        if new_y.shape[0] == 1 and not iter % 1:
            if ground_truth is not None:
                current_regret = (ground_truth - Y_queried.max().squeeze()).item()
            else:
                current_regret = 0.0
            print(
                f"Seed{seed} iter:{iter} "
                f"Candidate:{candidates.long().squeeze().tolist()}, "
                f"obj:{new_y.item():.4f} "
                f"threshold: {threshold_obj.item():.4f} "
                f"train_best:{new_best_obj.item():.4f} "
                f"queried_best:{queried_best_obj.item():.4f} "
                f"regret:{current_regret:.4f} "
                f"distance:{distance_from_start} "
                f"#Training:{len(Y_train)} "
                f"#Queried:{len(Y_queried)} "
                f"#Explored:{num_explored_combonodes} Time:{wall_time[-1]:.1f}s"
            )
        
        # update the trust region state, if applicable
        if use_trust_region:
            trust_region_state = update_state(state=trust_region_state, Y_next=new_y)
            if trust_region_state.shrink_triggered and not trust_region_state.restart_triggered:
                #n_hop = max(1, max_radius - trust_region_state.shrink_counter)
                n_hop = max(1, n_hop - 1 )
                ComboSubgraph = nx.ego_graph(ComboSubgraph, tuple(center.int().tolist()), n_hop)
                print(f" --- Shrink triggered: Combo-subgraph radius reduces to {n_hop} "
                    f"with {ComboSubgraph.number_of_nodes()} nodes --- ")
                # Only includes queried locations inside the current combosubgraph for traininig.
                idx_keep = torch.tensor(list(ComboSubgraph.nodes))
                X_train, Y_train = prune_baseline(X_queried, Y_queried, idx_keep)
                # As the context graph has changed, we re-compute the eigenbasis for the next BO iteration.
                # mapping and remapping dictionaries
                tuple_to_int, int_to_tuple = Index_Mapper(ComboSubgraph)  
                # Recompute the eigenbasis
                cached_eigenbasis = None 
                # Recompute the prior for X
                if use_prior:
                    X_prior = get_X_prior(ComboSubgraph, tuple(center.int().tolist()))
                # if use diffusion ard then update its order
                if label == "diffusion_ard":  
                    model_configurations["order"] = len(ComboSubgraph.nodes())
                trust_region_state.shrink_triggered = False

        # Recompute the subgraph at new centre if best location changes
        if (
            label in graph_kernels + ["local_search"]
            and (
                new_best_obj > (threshold_obj + 1e-3 * math.fabs(threshold_obj)) 
                or ComboSubgraph is None
            )
        ):
            print(f"------- Context ComboSubgraph Center Changed --------")
            best_idx = Y_train.argmax().cpu()
            best_loc = X_train[best_idx]
            center = best_loc if not exploitation else torch.tensor(start_combonode)
            # threshold_obj = threshold_obj if ( (exploitation or problem_kwargs["graph_type"]=='Road') and ComboSubgraph is None) else new_best_obj
            threshold_obj = (
                threshold_obj
                if (exploitation and ComboSubgraph is None)
                else new_best_obj
            )
            start_time = time()
            if new_best_obj > threshold_obj + 1e-3 * math.fabs(threshold_obj):
                n_contextsubgraph += 1
            if label in graph_kernels:
                ComboSubgraph, n_hop = ComboSubgraph_Constructor(
                    Problem.underlying_graph,
                    center.int().numpy().reshape(1, -1),
                    nx.Graph(),
                    Q=Q,
                    l_max=max_radius,
                    fast_computation=fast_computation,
                )
                # Keep tracking the total explored combo graph
                ComboGraph.add_edges_from(list(ComboSubgraph.edges))  
                # Only includes queried locations inside the current combosubgraph for traininig.
                idx_keep = torch.tensor(list(ComboSubgraph.nodes))
                X_train, Y_train = prune_baseline(X_queried, Y_queried, idx_keep)
                # As the context graph has changed, we re-compute the eigenbasis for the next BO iteration.
                # mapping and remapping dictionaries
                tuple_to_int, int_to_tuple = Index_Mapper(ComboSubgraph)  
                # Recompute the eigenbasis
                cached_eigenbasis = None 
                # Recompute the prior for X
                if use_prior:
                    X_prior = get_X_prior(ComboSubgraph, tuple(center.int().tolist()))
                # if use diffusion ard then update its order
                if label == "diffusion_ard":  
                    model_configurations["order"] = len(ComboSubgraph.nodes())
            # track the degree
            combosubgraph_center_degree = len(
                ComboNeighbors_Generator(Problem.underlying_graph, center))  
            print(f"time for subgraph construction: {(time()-start_time):.1f}s")

        # Record the experiment details for this iteration
        iter += 1  # counting the iterations, note usually it will be smaller than len(X_queried)

        exploration_list.append(num_explored_combonodes)
        degree_list.append(combosubgraph_center_degree)
        distance_list.append(distance_from_start)

    print(f"time for search: {time()-sss:.1f}s, n_restart: {n_restart}, "
        f"n_contextsubgraph: {n_contextsubgraph}")

    # ------------------ Save the Final Output --------------------
    if ground_truth is not None:
        regret = ground_truth - Y_queried.cpu()
    else:
        regret = None
    output_dict = {
        "label": label,
        "X": X_queried.cpu(),
        "Y": Y_queried.cpu(),
        "wall_time": wall_time,
        "regret": regret,
        "n_restart": n_restart,
        "distance": distance_list,
        "n_explored": exploration_list,
        "degree_list": degree_list,
    }
    with open(os.path.join(save_path, f"{str(seed).zfill(4)}_{label}.pt"), "wb") as fp:
        torch.save(output_dict, fp)
