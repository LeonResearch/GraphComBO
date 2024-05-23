import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool
from problems.GNN.args import parse_args
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

# This is the code to train the GNN and store its parameters
if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = TUDataset(root='data/TUDataset', name='DD')
    dataset = dataset.shuffle()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    gnn_args = parse_args()
    gnn_args = setup('./config.yaml')
    __import__("pdb").set_trace() 
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    hidden_dim = gnn_args.hidden_dim
    num_layers = gnn_args.num_layers
    num_epochs = gnn_args.num_epochs
    lr = gnn_args.lr

    model = GIN(num_features, num_classes, hidden_dim, num_layers)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training function
    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x.float(), data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_loader.dataset)

    # Evaluation function
    def train_acc():
        model.eval()
        correct = 0
        for data in train_loader:
            out = model(data.x.float(), data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(train_loader.dataset)
    
    acc_list = []
    for epoch in range(num_epochs):
        loss = train()
        acc = train_acc()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}')
        if len(acc_list) and acc > max(acc_list): # Here we only record the epoch with best training acc
            torch.save(model.state_dict(), './problems/GNN/gin_model_state.pt')
            print(f'model state saved at Epoch {epoch:03d} with training acc {acc:.4f}')
        acc_list.append(acc)
