import numpy as np
import networkx as nx
import torch
from typing import Optional, Dict, Any
from time import time
from search.trust_region import restart
from search.models import initialize_model, get_acqf, optimize_acqf, filter_invalid


def GraphBO_Search(
    Problem: object,
    context_graph: nx.Graph,
    label: str,
    X_queried: torch.Tensor,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    k: int,
    Q: int,
    anchor: torch.Tensor,
    n_initial_points: int,
    seed: int,
    batch_size: int,
    converge: bool,
    model_configurations: dict,  # experiment configurations
    tuple_to_int_mapper: dict,
    int_to_tuple_mapper: dict,  # the local-global index remappers
    cached_eigenbasis,
    trust_region_state,
    trust_region_kwargs,  # trust-region state variable
    acqf_kwargs,
    acqf_optim_kwargs,  # acqusition function settings
    model_optim_kwargs,
):
    if not trust_region_state.restart_triggered:
        # remap X_train to the new context subgraph
        X_mapped = tuple_to_int_mapper(X_train)  
        # 1. Build the surrogate model with pre-specified kernel choice.
        model, mll, cached_eigenbasis = initialize_model(
            train_X=X_mapped,
            train_Y=Y_train,
            context_graph=context_graph,
            covar_type=model_configurations["covar_type"],
            covar_kwargs={
                "order": model_configurations["order"]
            },  ## No order means context graph size
            ard=model_configurations["ard"],
            fit_model=True,
            cached_eigenbasis=cached_eigenbasis,
            optim_kwargs=model_optim_kwargs,
        )
        # 2. Define the acquisition function
        acq_func = get_acqf(
            model,
            X_baseline=X_mapped,
            train_Y=Y_train,
            batch_size=batch_size,
            acq_type=acqf_kwargs["type"],
        )
        # 3. Optimise the Acquisition function
        raw_candidates = optimize_acqf(
            acq_func,
            context_graph=context_graph,
            method="enumerate",
            batch_size=batch_size,
            X_avoid=X_mapped,
            acq_type=acqf_kwargs["type"],
            **acqf_optim_kwargs,
        )
        # Resart if all the nodes in the context subgraph have been queried
        if raw_candidates is None:  
            trust_region_state.restart_triggered = True
            candidates = None
            print("***** All ComboNodes in the ComboSubgraph are queried! *****")
        else:
            # the candidates labels are in terms of the local graph 
            # map them back to the labels of the "global" graph
            candidates = int_to_tuple_mapper(raw_candidates)
    else:  # when a restart is triggered
        print(f"========== Failtol Reached & Restart Triggered ========")
        # randomly respawn when all nodes in the subgraph are queried
        anchor = None if converge else anchor 
        candidates, trust_region_state = restart(  # Initial queried locations
            base_graph=Problem.underlying_graph,
            n_init=n_initial_points,
            seed=seed,
            k=k,
            context_subgraph_size=Q,  # Initial subgraph size
            X_avoid=X_queried,
            anchor=anchor,
            use_trust_region=True,
            options=trust_region_kwargs,
        )
        context_graph = None  # reset the context graph to be re-initialized later
        X_train = torch.zeros(0, X_train.shape[1]).to(X_train)
        Y_train = torch.zeros(0, 1).to(Y_train)
        print(candidates.long())
    return (
        context_graph,
        X_train,
        Y_train,
        candidates,
        trust_region_state,
        cached_eigenbasis,
    )
