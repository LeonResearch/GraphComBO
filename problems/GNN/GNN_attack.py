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
from utils.config_utils import setup

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dim, num_gc_layers):
        super(GIN, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lin = torch.nn.Linear(dim*num_gc_layers, dim)
        self.lin_final = torch.nn.Linear(dim, num_classes)

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)
            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        x = F.relu(self.lin(x))
        x = self.lin_final(x)
        return F.log_softmax(x, dim=-1)

def generate_prediction(graph, num_classes, gnn_args):
    model = GIN(graph.num_features, num_classes, gnn_args.hidden_dim, gnn_args.num_layers)
    model.load_state_dict(torch.load('./problems/GNN/gin_model_state.pt'))
    model.eval()
    loader = DataLoader([graph], batch_size=1, shuffle=False)
    pred = model(graph.x, graph.edge_index, graph.batch) # the original output is log-softmax
    pred = torch.exp(pred).squeeze().detach().cpu()
    return pred
