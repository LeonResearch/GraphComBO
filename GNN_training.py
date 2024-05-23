import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool
from problems.GNN.GNN_attack import GIN
from utils.config_utils import setup

# This is the code to train the victim GNN (a GIN model) and store its parameters
# it can achieve 96% training accuracy when using the pre-specified hyper-parameters
if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    gnn_args = setup('./problems/GNN/config.yaml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    dataset = TUDataset(root='./problems/TUDataset', name=gnn_args.dataset)
    dataset = dataset.shuffle()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    hidden_dim = gnn_args.hidden_dim
    num_layers = gnn_args.num_layers
    num_epochs = gnn_args.num_epochs
    lr = gnn_args.lr

    model = GIN(num_features, num_classes, hidden_dim, num_layers).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training function
    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
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
            data=data.to(device)
            out = model(data.x.float(), data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(train_loader.dataset)
    
    # Run training
    acc_list = []
    for epoch in range(num_epochs):
        loss = train()
        acc = train_acc()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}')
        if len(acc_list) and acc > max(acc_list): # Here we only record the epoch with best training acc
            #torch.save(model.state_dict(), './problems/GNN/gin_model_state.pt')
            print(f'model state saved at Epoch {epoch:03d} with training acc {acc:.4f}')
        acc_list.append(acc)
