import gpytorch.settings
import networkx as nx
from typing import Union, Tuple, Callable, Optional
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from botorch.cross_validation import gen_loo_cv_folds, CVFolds
from search.self_combograph import NeighborFinder


def eigendecompose_laplacian(
    context_graph: nx.Graph,
    dtype: torch.dtype = torch.float,
    normalized_laplacian: bool = True,
    normalized_eigenvalues: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform eigen-decomposition of ``context_graph``.
        We either take:
            a) a networkx graph.
    Note that the input graphs have to be directed to get a symmetric Laplacian and purely
        real eigenvalues
    Returns a tuple of torch.Tensor of shape ``N`` -> eigenvalues and ``N x N`` eigenvectors
    """
    if normalized_laplacian:
        L = nx.normalized_laplacian_matrix(context_graph).todense()
        if normalized_eigenvalues:
            # eigenvalues of normalized Laplacian are bounded by [0, 2].
            L /= 2.0  # divide by 2 to ensure the eigenvalues are between [0, 1]
    else:
        L = nx.laplacian_matrix(context_graph).todense()
    L = torch.from_numpy(L).to(dtype)
    eigenvals, eigenvecs = torch.linalg.eigh(
        L,
    )
    return eigenvals, eigenvecs


def fit_gpytorch_model(
    mll,
    model,
    train_x,
    train_y,
    train_iters: int = 100,
    lr: float = 0.1,
    print_interval: int = -1,
    return_loss: bool = False,
):
    with gpytorch.settings.debug(False):
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        model.likelihood.train()
        for i in range(train_iters):
            optimizer.zero_grad()  # Zero gradients from previous iteration
            output = model(train_x)  # Output from model
            loss = -mll(output, model.train_targets)  # Calc loss and backprop gradients
            if loss.ndim > 0:
                loss = loss.sum()
            loss.backward()
            optimizer.step()

    if return_loss:
        return model, loss.item()
    return model


def prune_baseline(
    X_queried: torch.Tensor, Y_queried: torch.Tensor, index_to_keep: torch.Tensor
):
    # first unsqueeze & reshape X_queried from (n, k) to (n, 1, k) then expand the
    # second dimension to the number of rows in index_to_keep
    X_q_exp = (
        X_queried.unsqueeze(-1)
        .reshape(-1, 1, X_queried.shape[-1])
        .expand(-1, index_to_keep.shape[0], -1)
    )
    # next compare if the corresponding element are the same and
    # use torch.all with torch.any to locate the common combonodes
    train_idx = (X_q_exp == index_to_keep).all(dim=-1).any(dim=-1)
    # Finally we use these indices as the training set
    return (
        X_queried[train_idx],
        Y_queried[train_idx],
    )  # corresponding to X_train, Y_train


def filter_invalid(X: torch.Tensor, X_avoid: torch.Tensor):
    """Remove all occurences of `X_avoid` from `X`."""
    X_set = {tuple(row.tolist()) for row in X}
    X_avoid_set = {tuple(row.tolist()) for row in X_avoid}
    ret = list(X_set - X_avoid_set)
    return torch.tensor(ret)


# Map the indices of nodes on combinatorial graph to numerical indices
def Index_Mapper(
    Graph,
):  # e.g. (2,8,13)-> 3, we need numerical indices for kernel computation
    node_list = [i for i in Graph.nodes()]
    mapping = {
        tuple_node: int_node for int_node, tuple_node in enumerate(node_list, start=0)
    }
    remapping = {v: k for k, v in mapping.items()}
    return mapping, remapping


def ComboNeighbors_Generator(
    underlying_graph: nx.Graph,  # the original underlying graph
    center_combonode: torch.Tensor,  # the center combo-node
    X_avoid: torch.Tensor = None,  # queried combo-nodes
    stochastic: bool = False,
):
    X = center_combonode.int().tolist()  # make combo-node to a int list
    G = nx.Graph()  # initialise an empty graph
    G.add_edges_from(
        NeighborFinder(underlying_graph, X)
    )  # get the 1-hop combo-neighbors
    neighbors = torch.tensor(list(G.neighbors(tuple(X))))
    if X_avoid is not None:
        neighbors = filter_invalid(neighbors, X_avoid)
    if stochastic and len(neighbors):
        return random.choice(neighbors).reshape(1, -1)
    return neighbors


def generate_start_location(
    underlying_graph: nx.Graph,  # the original graph
    k: int,  # number of elements in the combination
    method: str,
):  # the method we use to generate initial location
    if method == "ei":
        feature_dict = nx.eigenvector_centrality(underlying_graph)
    elif method == "betweenness":
        feature_dict = nx.betweenness_centrality(underlying_graph)
    elif method == "degree":
        feature_dict = nx.degree_centrality(underlying_graph)
    elif method == "pagerank":
        feature_dict = nx.pagerank(underlying_graph)
    feature = torch.tensor(list(feature_dict.values()))
    feature_sorted, idx = feature.sort()
    return idx[-k:].unsqueeze(0)


def graph_relabel(graph):
    node_list = [i for i in graph.nodes()]
    mapping = {old_node: new_idx for new_idx, old_node in enumerate(node_list, start=0)}
    inverse_mapping = {v: k for k, v in mapping.items()}
    return nx.relabel_nodes(graph, mapping), mapping, inverse_mapping


def get_X_prior(ComboSubgraph, center):
    X_hops = torch.tensor(list(dict(nx.shortest_path_length(ComboSubgraph, 
                                                             source=center)).values()))
    h_weight = 1/X_hops.max() * np.arange(1, X_hops.max()+1)
    X_prior = torch.exp(1/X_hops.max() * X_hops) / torch.sum(torch.exp(h_weight), dim=0)
    return X_prior


def prior_decay(max_iter, time_to_half, current_iter):
    constant = max_iter/time_to_half * torch.log(torch.tensor(0.5))/max_iter
    return torch.exp(constant * current_iter)
