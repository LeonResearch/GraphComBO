import numpy as np
import pickle as pkl
import random
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import to_networkx, from_networkx 
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from problems.GNN.GNN_attack import GIN, generate_prediction
from utils.config_utils import setup

if __name__ == '__main__':
    idx_graph_to_attack = 0
    gnn_args = setup('./problems/GNN/config.yaml')
    dataset = TUDataset(root='./problems/TUDataset', name=gnn_args.dataset)
    victim_graph = dataset[idx_graph_to_attack]
    print(victim_graph)
    graph = to_networkx(victim_graph).to_undirected()
    with open('./problems/GNN/victim_graph.pickle', 'wb') as path:
        pkl.dump(graph, path)
    # generate prediction for the original graph
    pred_base = generate_prediction(victim_graph, dataset.num_classes, gnn_args)
    # delete the k edges
    perturbed_graph = victim_graph.clone()
    perturbed_graph.edge_index = perturbed_graph.edge_index[:,:-2]
    pred_attack = generate_prediction(perturbed_graph, dataset.num_classes, gnn_args)

    wd = wasserstein_distance(pred_attack, pred_base)
    js = distance.jensenshannon(pred_attack, pred_base)
    print(f"is the original prediction correct: {pred_base.argmax() == victim_graph.y}")
    print(f"victim graph prediction: {pred_base.tolist()}")
    print(f"perturbed graph prediction: {pred_attack.tolist()}")
    print(f"WD: {wd:.4f}")
    print(f'JS: {js:.4f}')
    __import__("pdb").set_trace()
