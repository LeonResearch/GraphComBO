import numpy as np
import networkx as nx
import torch
from dataclasses import dataclass
from search.trust_region import restart
from search.models import initialize_model, get_acqf, optimize_acqf, filter_invalid


@dataclass
class GraphComBO:
    Graph: nx.Graph # The original graph
    ComboSubgraph: nx.Graph # The combinatorial subgraph
    kernel: str # Type of kernel 
    k: int # Size of the subset
    Q: int # Size of the subgraph
    anchor: torch.Tensor # The starting/restarting location
    n_init: int
    seed: int
    converge: bool
    noisy: bool
    tuple_to_int_mapper: dict
    int_to_tuple_mapper: dict
    kernel_configurations: dict
    cached_eigenbasis: torch.Tensor
    trust_region_state: object
    trust_region_kwargs: dict 
    acqf_kwargs: dict
    acqf_optim_kwargs: dict
    model_optim_kwargs: dict


    def get_candidates(self, X_train, Y_train, X_queried, X_prior=None):
        if not self.trust_region_state.restart_triggered:
            # remap X_train to the new context subgraph
            X_mapped = self.tuple_to_int_mapper(X_train)  
            # 1. Build the surrogate model with pre-specified kernel choice.
            model, mll, self.cached_eigenbasis = initialize_model(
                train_X=X_mapped,
                train_Y=Y_train,
                context_graph=self.ComboSubgraph,
                covar_type=self.kernel_configurations["covar_type"],
                covar_kwargs={"order": self.kernel_configurations["order"]}, 
                ard=self.kernel_configurations["ard"],
                fit_model=True,
                cached_eigenbasis=self.cached_eigenbasis,
                optim_kwargs=self.model_optim_kwargs,
            )
            # 2. Define the acquisition function
            acq_func = get_acqf(
                model,
                X_baseline=X_mapped,
                train_Y=Y_train,
                acq_type=self.acqf_kwargs["type"],
            )
            # 3. Optimise the Acquisition function
            raw_candidates, best_of_posterior_mean = optimize_acqf(
                model,
                acq_func,
                context_graph=self.ComboSubgraph,
                method="enumerate",
                noisy=self.noisy,
                X_avoid=X_mapped,
                X_prior=X_prior,
                acq_type=self.acqf_kwargs["type"],
                **self.acqf_optim_kwargs,
            )
            # Resart if all the nodes in the context subgraph have been queried
            if raw_candidates is None:  
                self.trust_region_state.restart_triggered = True
                candidates = None
                print("***** All ComboNodes in the ComboSubgraph are queried! *****")
            else:
                # the candidates labels are in terms of the local graph 
                # map them back to the labels of the "global" graph
                candidates = self.int_to_tuple_mapper(raw_candidates)
        else:  # when a restart is triggered
            print(f"========== Failtol Reached & Restart Triggered ========")
            # randomly respawn when all nodes in the subgraph are queried
            self.anchor = None if self.converge else self.anchor 
            candidates, self.trust_region_state = restart(  # Initial queried locations
                base_graph=self.Graph,
                n_init=self.n_init,
                seed=self.seed,
                k=self.k,
                context_subgraph_size=self.Q,  # Initial subgraph size
                X_avoid=X_queried,
                anchor=self.anchor,
                use_trust_region=True,
                options=self.trust_region_kwargs,
            )
            self.ComboSubgraph = None  # reset the context graph to be re-initialized later
            X_train = torch.zeros(0, X_train.shape[1]).to(X_train)
            Y_train = torch.zeros(0, 1).to(Y_train)
            print(candidates.long())
        return candidates, X_train, Y_train, self.ComboSubgraph
