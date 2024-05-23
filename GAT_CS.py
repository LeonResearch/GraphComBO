import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, Coauthor
from torch_geometric.nn import GINConv, global_add_pool, GATConv
from problems.GNN.GNN_attack import GIN
from utils.config_utils import setup

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.5)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, dropout=0.5)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# This is the code to train the victim GNN (a GIN model) and store its parameters
# it can achieve 96% training accuracy when using the pre-specified hyper-parameters
if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    dataset = Coauthor(root='./problems/Coauthor', name='CS')
    data = dataset[0] 
    data=data.to(device)
    #dataset = dataset.shuffle()
    #train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    hidden_dim = 64
    num_epochs = 200
    lr = 0.01

    model = GAT(dataset.num_features, hidden_channels=hidden_dim, out_channels=dataset.num_classes, 
                num_heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training function
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        return loss.item()


    # Evaluation function
    def train_acc():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = int((pred == data.y).sum())
        return correct / len(data.y), out
    
    # Run training
    acc_list = []
    for epoch in range(num_epochs):
        loss = train()
        acc, emb = train_acc()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}')
        if len(acc_list) and acc > max(acc_list): # Here we only record the epoch with best training acc
            torch.save(model.state_dict(), './problems/GNN/gat_cs_state.pt')
            torch.save(emb, './problems/GNN/gat_cs_emb.pt')
            print(f'model state saved at Epoch {epoch:03d} with training acc {acc:.4f}')
        acc_list.append(acc)
