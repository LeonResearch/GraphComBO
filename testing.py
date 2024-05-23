import torch
import torch.nn as nn


h = 5

def get_Qh(h):
    f = nn.Softmax()
    Q_h = 1/h * torch.arange(1,h+1).to(torch.float)
    return f(Q_h)

for h in range(1,10):
    x = get_Qh(h)
    print(f'h is {h} and x is {x}')


__import__('pdb').set_trace()
