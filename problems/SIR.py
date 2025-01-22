import numpy as np
import networkx as nx
import multiprocessing
import time
import torch
import warnings
import future.utils
import ndlib.models.ModelConfig as mc
from ndlib.models.epidemics.SIRModel import SIRModel

def get_initial_status(Graph, ComboNode, mapping): # set the initial state to the chosen ComboNode
    node_list = [i for i in Graph.nodes()]
    for node in node_list:
        if node in ComboNode:
            mapping[node] = 2
    return mapping

def One_SIR_Run(graph, cfg, ComboNode, T, idx):
    model = SIRModel(graph, idx)
    model.set_initial_status(cfg)
    model.status = get_initial_status(graph, ComboNode, model.status)
    iterations = model.iteration_bunch(T)
    return iterations

def SIR_MC(T,N,n_samples,threshold,parallel_function,parallel=True):
    # Simulation execution
    if parallel:
        with multiprocessing.Pool() as pool:
            results = pool.map(parallel_function, list(range(n_samples)))
    else:
        results = list(map(parallel_function,list(range(n_samples))))
    infection_time_list = []
    for j in range(n_samples):
        S = [results[j][i]['node_count'][0] for i in range(T)]
        infection_time = sum(np.array(S)>(N*(1-threshold)))
        infection_time_list.append(infection_time)
    infection_time_list = np.array(infection_time_list)
    infection_time_valid = infection_time_list[~(infection_time_list == T)] # filter out invalid simulations
    function_value = np.mean(infection_time_valid).round(1)
    return function_value/T # optional: make it negative if we want to minimize it in the experiment.

def individual_infection_time(model, iter_max, seed):
    # Keep track of nodes that have been infected
    set_infected = set([node for node, nstatus in future.utils.iteritems(model.status) if nstatus == model.available_statuses['Infected']])
    set_susceptible = [node for node, nstatus in future.utils.iteritems(model.status) if nstatus == model.available_statuses['Susceptible']]

    # Get parameters
    epsilon = 0.001 # Spontaneous infection
    
    iteration = model.iteration()
    feature = iteration['status'] # Get initial starting points for infected nodes
    
    while (iteration['node_count'][1] != 0) and (iteration['iteration'] < iter_max):
        iteration = model.iteration()
        current_status = iteration['status']
        
        # Introduce spontaneous infections
        n_susceptible = len(set_susceptible)
        n_drawn = int(np.random.binomial(n=n_susceptible, p=epsilon, size=1))
        list_spontaneous_infection = np.random.RandomState(seed).choice(
            set_susceptible, n_drawn, replace=False
        ).tolist()
        for spontaneous_infection in list_spontaneous_infection:
            model.status[spontaneous_infection] = 1
        
        # Add them to new infections and value function
        for key in list_spontaneous_infection:
            if key not in set_infected:
                set_infected.add(key)
                feature[key] = iteration['iteration'] + 1
        for key, value in current_status.items():
            if value == 1 and key not in set_infected:
                set_infected.add(key)
                feature[key] = iteration['iteration'] + 1 
    '''
    for key, value in feature.items():
        if value != 0:
            feature[key] = iteration['iteration'] - feature[key]
    '''
    for key, value in feature.items():
        if value != 0:
            feature[key] = (1 - (feature[key] - 1) /
                            (iteration['iteration'] + 1))**2
    return torch.tensor(list(feature.values())).to(torch.float)
