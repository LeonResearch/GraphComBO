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
from search.baselines import DFS_BFS_Search, Local_Search
from search.graphbo import GraphBO_Search
from search.self_combograph import ComboSubgraph_Constructor
from search.utils import (
    prune_baseline,
    ComboNeighbors_Generator,
    generate_start_location,
    eigendecompose_laplacian,
    filter_invalid,
    Index_Mapper,
)
from search.trust_region import update_state, restart


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
    large_Q: bool = False,  # if True we use a faster way to compute the ComboSubgraph
    exploitation: bool = False,
    k: int = 2,  # Number of combinations
    acqf_kwargs: Optional[dict] = None,
    acqf_optim_kwargs: Optional[dict] = None,
    model_optim_kwargs: Optional[Dict[str, Any]] = None,
    trust_region_kwargs: Optional[Dict[str, Any]] = None,
    problem_kwargs: Optional[Dict[str, Any]] = None,
    dtype: torch.dtype = torch.float,
    device: str = "cpu",
    save_frequency: int = 1,
    animation: bool = False,
    animation_interval: int = 20,
    order=None,
):
    graph_kernels = [
        "polynomial",
        "polynomial_suminverse",
        "diffusion",
        "diffusion_ard",
    ]
    trust_region_kwargs = trust_region_kwargs or {}
    problem_kwargs = problem_kwargs or {}

    """ 
    # we don't need to set random seeds since we repeated each experiment 
    # for 50 times in our paper.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """
    save_path = os.path.join(save_path, label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tkwargs = {"dtype": dtype, "device": device}
    acqf_optim_kwargs = acqf_optim_kwargs or {}
    acqf_kwargs = acqf_kwargs or {}
    acqf_kwargs["type"] = "ucb" if exploitation else "ei"
    model_optim_kwargs = model_optim_kwargs or {}

    # ------------- Initialise the Problem --------------
    Problem = get_synthetic_problem(
        problem_name, seed=seed, problem_kwargs=problem_kwargs
    )
    print(
        f"Using {label} model with Q={Q} radius={max_radius} failtol={trust_region_kwargs['fail_tol']} "
        f"start={start_location_method} restart={restart_location_method} exploitation={exploitation}"
    )
    ground_truth = (
        Problem.ground_truth.cpu() if Problem.ground_truth is not None else None
    )
    use_trust_region = label in graph_kernels
    n_restart = 0
    n_contextsubgraph = 0
    large_Q = True if (k >= 20 and len(Problem.underlying_graph) >= 1000) else large_Q

    if start_location_method in [
        "ei",
        "betweenness",
        "degree",
        "pagerank",
    ]:  # useful when k is large
        start_location = generate_start_location(
            Problem.underlying_graph, k, start_location_method
        )
    else:
        start_location = None

    candidate_start, trust_region_state = restart(  # Initial queried locations
        base_graph=Problem.underlying_graph,
        n_init=1,
        seed=seed,
        k=k,
        batch_size=batch_size,
        context_subgraph_size=Q,  # Initial subgraph size
        X_avoid=None,
        anchor=start_location,
        n_restart=n_restart,
        iterations=iterations,
        initialization=True,
        use_trust_region=use_trust_region,
        options=trust_region_kwargs,
    )

    candidate_neighbours = ComboNeighbors_Generator(
        Problem.underlying_graph, candidate_start.squeeze()
    )  # track the degree
    n_init_actual = min(n_initial_points, len(candidate_neighbours))
    candidates_idx = torch.from_numpy(
        np.random.RandomState(seed).choice(
            np.arange(candidate_neighbours.shape[0]), n_initial_points, replace=False
        )
    )
    candidates = torch.cat(
        [candidate_start, candidate_neighbours[candidates_idx]], dim=0
    )
    X_queried = candidates.to(
        **tkwargs
    )  # X_queried/Y_queried is the set of queried nodes/values.
    Y_queried = Problem(X_queried).to(
        **tkwargs
    )  # X_queried is the global indices (i.e. on the original graph).
    X_train = (
        X_queried.clone()
    )  # X_train, Y_train are the training sets, i.e. nodes inside subgraph at each iter.
    Y_train = (
        Y_queried.clone()
    )  # They will be updated at each iteration by "prune_baseline".
    # Note X_train/Y_train are useless for the baselines, they are just the same as X_queried/Y_queried.

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
    Q = max(int(combosubgraph_center_degree / 4), Q) if exploitation else Q
    if acqf_optimizer is None:
        acqf_optimizer = "enumerate" if Problem.problem_size <= 1000 else "local_search"
    # --------------------- Initialise the Models --------------------
    # initialise BO models
    if label in graph_kernels:
        ComboSubgraph, n_hop = ComboSubgraph_Constructor(
            Problem.underlying_graph,
            torch.tensor(start_combonode).numpy().reshape(1, -1),
            nx.Graph(),
            Q=300,
            l_max=max_radius,
            large_Q_computation=large_Q,
        )
        all_nodes = torch.tensor(list(ComboSubgraph.nodes))
        y = torch.zeros(len(all_nodes))
        for i in range(len(all_nodes)):
            print(f"calculating {i}th node")
            y[i] = Problem(all_nodes[i])
        # with open(os.path.join(save_path, f"y.pt"), "wb") as fp:
        with open(f"y.pt", "wb") as fp:
            torch.save(y, fp)
        import pickle

        pickle.dump(ComboSubgraph, open("combosubgraph.pickle", "wb"))
        __import__("pdb").set_trace()
        ComboGraph = nx.Graph()
        ComboGraph.add_edges_from(
            list(ComboSubgraph.edges)
        )  # Keep tracking the total explored combo graph
        X_prior = (
            1
            / (
                torch.exp(
                    torch.tensor(
                        list(
                            dict(
                                nx.shortest_path_length(
                                    ComboSubgraph, source=start_combonode
                                )
                            ).values()
                        )
                    )
                )
            )
            if exploitation
            else None
        )

        # Only include qu8ried nodes that are inside ComboSubgraph for training
        idx_keep = torch.tensor(list(ComboSubgraph.nodes))
        X_train, Y_train = prune_baseline(X_queried, Y_queried, idx_keep)

        # Functions to map between tuple node-label and int node-label
        # This is for kernels in graph GP, which will regard int node labels as "features",
        # and use these labels as "indices" to choose which eigenvectors to use.
        # i.e., when computing covariance between node i and node j, we need to use
        # engenvector_matrix[i] and eigenvector_matrix[j] in the kernel functions.
        tuple_to_int, int_to_tuple = Index_Mapper(
            ComboSubgraph
        )  # mapping and remapping dictionaries

        def tuple_to_int_mapper(x):  # map node labels from tuples to integers
            return (
                torch.tensor([tuple_to_int[tuple(i.int().tolist())] for i in x])
                .to(x)
                .reshape(-1, 1)
            )

        def int_to_tuple_mapper(x):
            return torch.tensor(  # map node labels from integers back to tuple
                [int_to_tuple[int(i)] for i in x]
            ).to(x)

        # Set up the configurations for surrogate models with different kernels
        model_configurations = {"covar_type": label, "order": None, "ard": True}
        if label in [
            "diffusion_ard"
        ]:  # both diffusion_ard and diffusion share the same kernel
            model_configurations[
                "covar_type"
            ] = "diffusion"  # the differences are "ard" and "order"
            model_configurations["order"] = len(
                ComboSubgraph.nodes
            )  ## Change to order size context graph
        elif label == "diffusion":
            model_configurations["ard"] = False  # Diffusion kernel without ARD
        elif label in ["polynomial_suminverse", "polynomial"]:
            model_configurations["order"] = n_hop  ## Change to order size context graph

    # Initialise baselines
    elif label in ["random", "dfs", "bfs", "local_search"]:
        ComboSubgraph = (
            nx.Graph() if label == "local_search" else None
        )  # local context combo-subgraph is not needed in baselines
        ComboGraph = nx.Graph()
        ComboGraph.add_nodes_from([start_combonode])
        num_explored_combonodes = 0
        if label in ["dfs", "bfs"]:  # Initialise the stacks for DFS/BFS
            list_stacks = []
            for i in range(batch_size):
                neighbors_current = ComboNeighbors_Generator(
                    Problem.underlying_graph,
                    torch.tensor(start_combonode),
                    X_avoid=X_queried,
                )
                neighbors_current_tuple = [tuple(x) for x in neighbors_current.tolist()]
                random.shuffle(
                    neighbors_current_tuple
                )  # make it random for multiple experiments
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
                comboedges_to_add = list(
                    tuple(map(tuple, i)) for i in comboedges_array_to_add
                )  # Convert np.array to list of tuples
                ComboGraph.add_edges_from(
                    comboedges_to_add
                )  # record the explored structure

                num_explored_combonodes = len(
                    ComboSubgraph_Constructor(
                        Problem.underlying_graph,
                        [start_combonode],
                        nx.Graph(),
                        Q=1e7,
                        l_max=1,
                        large_Q_computation=False,
                    )[0]
                )
                hop_tracker = 1

    # Set some experiment recorders
    cached_eigenbasis = None
    use_cached_eigenbasis = True  # If the subgraph doesn't change from last iter, then no need for eigendecomposition
    iter = 1
    sss = time()
    exploration_list = []
    degree_list = []
    distance_list = []
    # ------------------------ Search Starts -------------------------
    while len(X_queried) < iterations:
        start_time = time()
        anchor = (
            queried_best_loc.unsqueeze(0)
            if restart_location_method == "queried_best"
            else start_location
        )
        n_initial_points = 1
        # --------------- Select Query Location ------------------
        if label == "random":  # We use a random sampler similar to "restart"
            candidates = set()
            """
            candidate = torch.from_numpy(np.random.RandomState(
                (seed+1)*iterations + iter).choice(list(Problem.underlying_graph.nodes),
                                                   k,replace=False))
            """
            while len(candidates) < iterations:
                for patience in range(
                    iterations
                ):  # each time we sample "iteraions" candidates
                    # and later we filter out the invalid ones
                    # This is the version of sampling without random seed.
                    candidate = torch.from_numpy(
                        np.random.choice(
                            list(Problem.underlying_graph.nodes), k, replace=False
                        )
                    )
                    candidate = torch.sort(candidate)[0]
                    candidate = set([tuple(candidate.tolist())])
                    candidates = candidates.union(candidate)
                candidates = torch.tensor(list(candidates))
                candidates = filter_invalid(
                    candidates, X_queried
                )  # Filter out the visited nodes
                if candidates.shape[0] < iterations:
                    candidates = set([tuple(i.tolist()) for i in candidates])

            if len(candidates) > iterations:
                candidates = candidates[:iterations]
        elif label in ["bfs", "dfs"]:
            candidates, list_stacks, ComboGraph = DFS_BFS_Search(
                Problem,
                X_queried,
                label,
                list_stacks,
                seed,
                k,
                anchor,
                n_initial_points,
                batch_size,
                n_restart,
                iterations,
                ComboGraph,
            )
        elif label == "local_search":
            candidates, ComboGraph = Local_Search(
                Problem,
                X_queried,
                queried_best_loc,
                seed,
                k,
                anchor,
                n_initial_points,
                batch_size,
                n_restart,
                iterations,
                ComboGraph,
            )
        # Use Graph BO methods for query point selection
        elif label in graph_kernels:
            (
                ComboSubgraph,
                X_train,
                Y_train,
                candidates,
                trust_region_state,
                n_restart,
                cached_eigenbasis,
            ) = GraphBO_Search(
                Problem,
                ComboSubgraph,
                label,
                X_queried,
                X_train,
                Y_train,
                X_prior,
                k,
                Q,
                anchor,
                n_restart,
                n_initial_points,
                iterations,
                seed,
                batch_size,
                model_configurations,
                tuple_to_int_mapper,
                int_to_tuple_mapper,
                cached_eigenbasis,
                use_cached_eigenbasis,
                trust_region_state,
                trust_region_kwargs,
                acqf_kwargs,
                acqf_optim_kwargs,
                model_optim_kwargs,
            )

        # --------------- Evaluate Query Location -------------------
        if candidates is None:
            continue  # Skip the following section if restart is triggerd because all combo-nodes are queried.
        # do not append the candidate if it is a previously qeuried anchor point at restart
        if X_train.shape[0] == 0 and anchor is not None:
            new_y = start_combonode_Y  # Just simply obtain the initial location
        else:
            new_y = Problem(candidates)  # Query the selected location
            X_queried = torch.cat(
                [X_queried, candidates], dim=0
            )  # append the new queried point to the queried set
            Y_queried = torch.cat([Y_queried, new_y], dim=0)
        X_train = torch.cat([X_train, candidates], dim=0)
        Y_train = torch.cat(
            [Y_train, new_y], dim=0
        )  # append the new queried point to the training set
        new_best_obj = Y_train.max().squeeze().cpu().numpy()
        queried_best_obj = Y_queried.max().squeeze().cpu().numpy()
        queried_best_loc = X_queried[Y_queried.argmax().cpu()]

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
        # We also record the exploration size and the current combosubgraph center degree
        if label in ["bfs", "dfs"]:
            if distance_from_start > hop_tracker:
                print("hi")
                hop_tracker = distance_from_start
                combosubgraph_center_degree = len(
                    ComboNeighbors_Generator(
                        Problem.underlying_graph, candidates.squeeze()
                    )
                )  # track the degree
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
                            large_Q_computation=False,
                        )[0]
                    )

        # update the trust region state, if applicable
        if use_trust_region:
            trust_region_state = update_state(state=trust_region_state, Y_next=new_y)

        wall_time.append(time() - start_time)
        if new_y.shape[0] == 1 and not iter % 1:
            if ground_truth is not None:
                current_regret = (ground_truth - Y_queried.max().squeeze()).item()
            else:
                current_regret = 0.0
            print(
                f"Seed{seed} iter:{iter} "  # + f'Candidate:{candidates.squeeze().tolist()}, '
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

        # Recompute the subgraph at new centre if best location changes
        if (label in graph_kernels + ["local_search"]) and (
            new_best_obj > (threshold_obj + 1e-3 * math.fabs(threshold_obj))
            or ComboSubgraph is None
        ):
            print(f"------- Context ComboSubgraph Center Changed --------")
            best_idx = Y_train.argmax().cpu()
            best_loc = X_train[best_idx]
            center = best_loc if not exploitation else torch.tensor(start_combonode)
            threshold_obj = (
                threshold_obj
                if (
                    (exploitation or problem_kwargs["graph_type"] == "Road")
                    and ComboSubgraph is None
                )
                else new_best_obj
            )
            start_time = time()
            if new_best_obj > threshold_obj:
                n_contextsubgraph += 1
            if label in graph_kernels:
                ComboSubgraph, n_hop = ComboSubgraph_Constructor(
                    Problem.underlying_graph,
                    center.int().numpy().reshape(1, -1),
                    nx.Graph(),
                    Q=Q,
                    l_max=max_radius,
                    large_Q_computation=large_Q,
                )
                ComboGraph.add_edges_from(
                    list(ComboSubgraph.edges)
                )  # Keep tracking the total explored combo graph
                # Only includes queried locations inside the current combosubgraph for traininig.
                idx_keep = torch.tensor(list(ComboSubgraph.nodes))
                X_train, Y_train = prune_baseline(X_queried, Y_queried, idx_keep)
                # As the context graph has changed, we re-compute the eigenbasis for the next BO iteration.
                tuple_to_int, int_to_tuple = Index_Mapper(
                    ComboSubgraph
                )  # mapping and remapping dictionaries
                use_cached_eigenbasis = False
                if (
                    label == "diffusion_ard"
                ):  # if use diffusion ard then update its order
                    model_configurations["order"] = len(ComboSubgraph.nodes())
                # Set a prior for X_train in exploitation mode, which is a simple weight based on distance
                X_prior = (
                    1
                    / (
                        torch.exp(
                            torch.tensor(
                                list(
                                    dict(
                                        nx.shortest_path_length(
                                            ComboSubgraph,
                                            source=tuple(center.int().tolist()),
                                        )
                                    ).values()
                                )
                            )
                        )
                    )
                    if exploitation
                    else None
                )
            combosubgraph_center_degree = len(
                ComboNeighbors_Generator(Problem.underlying_graph, center)
            )  # track the degree
            print(f"time for subgraph construction: {(time()-start_time):.1f}s")
        else:  # just use the subgraph from last iteration
            use_cached_eigenbasis = True

        # Record the experiment details for this iteration
        iter += 1  # counting the iterations, note usually it will be smaller than len(X_queried)

        exploration_list.append(num_explored_combonodes)
        degree_list.append(combosubgraph_center_degree)
        distance_list.append(distance_from_start)

    print(
        f"time for search: {time()-sss:.1f}s, n_restart: {n_restart}, n_contextsubgraph: {n_contextsubgraph}"
    )

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
