from dataclasses import dataclass
import math
from typing import Any, Optional, Dict
import torch
import networkx as nx
import numpy as np
from search.utils import filter_invalid


@dataclass
class TrustRegionState:
    dim: int = (1,)
    n_nodes: int = 50
    n_nodes_min: int = 5
    n_nodes_max: int = 100
    failure_counter: int = 0
    shrink_counter: int = 0
    success_counter: int = 0
    fail_tol: int = float("nan")  # Note: Post-initialized
    succ_tol: int = 10  # Note: The original paper uses 3
    shrink_tol: int = 5
    best_value: float = -float("inf")
    restart_triggered: bool = False
    shrink_triggered: bool = False
    trust_region_multiplier: float = 1.5
    large_Q: bool = False


def update_state(
    state: "TrustRegionState",
    Y_next: torch.Tensor,
):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
        state.shrink_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    
    if state.failure_counter != 0 and state.failure_counter % state.shrink_tol == 0:  
        state.shrink_triggered = True # Shrink trust region
        state.shrink_counter += 1

    if state.success_counter == state.succ_tol:  # Expand trust region
        state.n_nodes = int(
            min(state.trust_region_multiplier * state.n_nodes, state.n_nodes_max)
        )
        state.success_counter = 0
    elif state.failure_counter == state.fail_tol:  # Restart triggered
        state.restart_triggered = True
        state.failure_counter = 0
        state.shrink_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.n_nodes < state.n_nodes_min:
        state.restart_triggered = True
    return state


def restart(
    base_graph: nx.Graph,
    n_init: int,  # initial points to be queried in the ComboSubgraph
    seed: int,
    k: int = 4,  # number of combinations
    context_subgraph_size: int = None,  # this is Q, size of the ComboSubgraph
    use_trust_region: bool = True,
    patience: int = 1000,
    initialization: bool = False,
    warmup: str = 'random',
    X_avoid: Optional[torch.Tensor] = None,
    anchor: Optional[torch.Tensor] = None,
    options: Optional[Dict[str, Any]] = None,
):
    """
    Restart function. Used at either at the initialization of optimization, or when
    a trust region restart is triggered.
    """
    # this is the kwargs options to initialize a new TrustRegionState object
    default_options = {
        "n_nodes_min": 5,
        "fail_tol": 20,
        "succ_tol": 10,
        "trust_region_multiplier": 1.5,
    }
    default_options.update(options or {})
    if anchor is not None:  # use the specified restart location
        candidates = anchor
    else:  # sample n_init random locations
        candidates = torch.zeros([0, k])
        iter = 0
        while len(candidates) < n_init:
            candidate_seeds = torch.from_numpy(
                np.random.RandomState(seed * patience + iter).choice(2**32, n_init)
            )  # generate random seeds first
            candidates = candidates if len(candidates) else torch.zeros([0, k])
            for seed_ in candidate_seeds:
                if initialization:
                    if warmup == 'random':
                        # use the same starting location at initialization for all methods
                        candidate = torch.from_numpy(
                            np.random.RandomState(seed_).choice(
                                list(base_graph.nodes), k, replace=False
                            )
                        )
                    elif warmup == 'random_walk':
                        if len(candidates):
                            current_location = candidates[-1]
                        else:
                            current_location = torch.from_numpy(
                                np.random.RandomState(seed_).choice(
                                    list(base_graph.nodes), k, replace=False
                                )
                            )
                        candidate = [
                            np.random.RandomState(seed_).choice(
                                list(base_graph.neighbors(
                                    current_location[i].item())))
                            for i in range(k)
                        ]
                        candidate = torch.tensor(candidate)

                else:  # sampling without random seeds during search
                    candidate = torch.from_numpy(
                        np.random.choice(list(base_graph.nodes), k, replace=False)
                    )
                # Important! sort the node label for unique indentification
                candidate = torch.sort(candidate)[0]  
                if not (candidate == candidates).all(-1).any(-1):
                    candidates = torch.vstack([candidates, candidate.reshape(1, -1)])
            if X_avoid is not None:  # Filter out the visited nodes
                candidates = filter_invalid(candidates, X_avoid)
            if iter >= patience:
                raise RuntimeError(
                    "Can not sample enough combo-nodes at restart with the current patience"
                )
            iter += 1
        if len(candidates) >= n_init:
            candidates = candidates[:n_init]
    # initialize a new state
    if use_trust_region:
        current_failtol = default_options["fail_tol"]
        current_tr = default_options["trust_region_multiplier"]
        trust_region_state = TrustRegionState(
            dim=1,
            n_nodes_max=context_subgraph_size,
            **default_options,
        )
        return candidates, trust_region_state
    return candidates, None
