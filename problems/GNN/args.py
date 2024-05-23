import argparse
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dataset', type=str, default='ENZYMES')
    args = parser.parse_args()
    return args
