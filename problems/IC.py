import numpy as np
import networkx as nx
import random
from functools import partial

def propagate(g,p,new_active):
    targets = []
    for node in new_active:
        targets += g.neighbors(node)
    return (targets)
def IC(graph,S,p,i): 
    new_active, A = S[:], S[:]
    idx = 0
    targets = []
    while new_active:
        # 1. Find out-neighbors for each newly active node
        targets = propagate(graph,p,new_active)
        # 2. Determine newly activated neighbors (set seed and sort for consistency)
        success = np.random.RandomState(i).uniform(0,1,len(targets)) < p
        new_ones = list(np.extract(success, sorted(targets)))
        # 3. Find newly activated nodes and add to the set of activated nodes
        new_active = list(set(new_ones) - set(A))
        A += new_active       
    return len(A)
def IC_MC(graph,S,p,mc):
    partial_func = partial(IC,graph,S,p)
    results = list(map(partial_func,list(range(mc))))
    return np.mean(results).round(1).item() / graph.number_of_nodes() * 100
