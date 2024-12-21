import numpy as np
import networkx as nx
import torch
import random
from dataclasses import dataclass
from search.utils import ComboNeighbors_Generator, filter_invalid
from search.trust_region import restart

@dataclass
class Baseline:
    Graph: nx.Graph 
    ComboGraph: nx.Graph
    k: int
    X_queried: torch.Tensor
    X_train: torch.Tensor
    Y_train: torch.Tensor
    seed: int
    n_init: int
    
    
    # =========== These baselines operate on the original graph ============
    def Random_Sample(self, n_samples, patience: int = 50000):
        iter = 0
        candidates = torch.zeros([0, self.k])
        while len(candidates) <= n_samples:
            candidates = candidates if len(candidates) else torch.zeros([0, self.k])
            candidate = torch.from_numpy(
                np.random.choice(list(self.Graph.nodes), self.k, replace=False)
            )
            # Important! sort the node label for unique indentification
            candidate = torch.sort(candidate)[0]  
            # Append this candidate if it is not in the candidate set.
            if not (candidate == candidates).all(-1).any(-1):
                candidates = torch.vstack([candidates, candidate.reshape(1, -1)])
            candidates = filter_invalid(candidates, self.X_queried)
            if iter >= patience:
                raise RuntimeError(
                    f"Can not sample enough combo-nodes at restart with the current patience! "
                    f"Consider increasing the combinatorial space on a larger graph or larger k."
                )
            iter += 1
        return candidates        


    def kRandom_Walk(self): 
        current_location = self.X_queried[-1].long()
        iter, patience = 0, 1000
        while iter < patience:
            # randomly choose a neighbor for each random walk
            # combine them together as the candidate to query next
            candidates = [
                random.choice(list(self.Graph.neighbors(current_location[i].item())))
                for i in range(self.k)
            ]
            candidates = torch.tensor(candidates).unsqueeze(0)
            candidates = filter_invalid(candidates, self.X_queried)
            if candidates.shape[0] == 1:
                return candidates
            else:
                iter += 1
        print("Can not sample enough nodes for the k-random walk, restarting at a new location ...")
        candidates, trust_region_state = restart(
            base_graph=self.Graph,
            n_init=self.n_init,
            seed=self.seed,
            k=self.k,
            X_avoid=self.X_queried,
            anchor=None,
            use_trust_region=False,
        )
        return candidates


    def kLocal_Search(self): # This is defined on the original graph
        best_loc = self.X_train[self.Y_train.argmax().cpu()].long()
        iter, patience = 0, 1000
        while iter < patience:
            # randomly choose a neighbor for each random walk
            # combine them together as the candidate to query next
            candidates = [
                random.choice(list(self.Graph.neighbors(best_loc[i].item()))) for i in range(self.k)
            ]
            candidates = torch.tensor(candidates).unsqueeze(0)
            candidates = filter_invalid(candidates, self.X_queried)
            if candidates.shape[0] == 1:
                return candidates, self.X_train, self.Y_train
            else:
                iter += 1
        print("Can not sample enough nodes for the k-local search, restarting at a new location ...")
        candidates, trust_region_state = restart(
            base_graph=self.Graph,
            n_init=self.n_init,
            seed=self.seed,
            k=self.k,
            X_avoid=self.X_queried,
            anchor=None,
            use_trust_region=False,
        )
        X_train = torch.zeros(0, self.X_train.shape[1]).to(self.X_train)
        Y_train = torch.zeros(0, 1).to(self.Y_train)
        return candidates, X_train, Y_train


    # ========= These baselines operate on the combinatorial graph ==========
    def DFS_BFS_Search(self, label, list_stacks):
        visited = set(tuple(row.int().tolist()) for row in self.X_queried)
        flag = 1
        for stack in list_stacks:
            flag *= len(stack)
        if flag:
            candidates = []
            for i_stack, stack in enumerate(list_stacks):
                element = stack.pop()
                if element not in visited:
                    neighbors_element = ComboNeighbors_Generator(
                        self.Graph, torch.tensor(element), X_avoid=self.X_queried
                    )
                    neighbors_element_tuple = [tuple(x) for x in neighbors_element.tolist()]
                    # make the algorithm random from different runs
                    random.shuffle(neighbors_element_tuple)
                    # Then do DFS or BFS
                    if label == "dfs":
                        stack = stack + neighbors_element_tuple
                    elif label == "bfs":
                        stack = neighbors_element_tuple + stack
                    list_stacks[i_stack] = stack

                    # Now we add these new graph structure to ComboGraph for recording purposes
                    if len(neighbors_element):
                        center_combonode_to_stack = np.repeat(
                            np.array(element).reshape(1, -1), len(neighbors_element), axis=0
                        )
                        comboedges_array_to_add = np.stack(
                            (neighbors_element, center_combonode_to_stack), axis=1
                        ).astype(int)
                        comboedges_to_add = list(
                            tuple(map(tuple, i)) for i in comboedges_array_to_add
                        )  # Convert np.array to list of tuples
                        # then record the explored structure
                        self.ComboGraph.add_edges_from(comboedges_to_add)  
                candidates.append(list(element))
            candidates = torch.tensor(candidates)
        else:  # Restard if stuck
            print(f"=========== Restart Triggered ==============")
            candidates, trust_region_state = restart(  # Initial queried locations
                base_graph=self.Graph,
                n_init=self.n_init,
                seed=self.seed,
                k=self.k,
                X_avoid=self.X_queried,
                anchor=None,
                use_trust_region=False,
            )
        return candidates, self.ComboGraph, list_stacks


    def Local_Search(self):
        best_loc = self.X_train[self.Y_train.argmax().cpu()].long()
        neighbors_of_best = ComboNeighbors_Generator(
            self.Graph, best_loc, X_avoid=self.X_queried
        )
        # when we cannot find a valid point for the local search, we have reached a local minimum.
        # randomly spawn a new starting point
        if not len(neighbors_of_best):  # restart if there is no neighbor around best_loc
            print(f"=========== Restart Triggered ==============")
            candidates, trust_region_state = restart(  # Initial queried locations
                base_graph=self.Graph,
                n_init=self.n_init,
                seed=self.seed,
                k=self.k,
                X_avoid=self.X_queried,
                anchor=None,
                use_trust_region=False,
            )
            X_train = torch.zeros(0, self.X_train.shape[1])
            Y_train = torch.zeros(0, 1)
            return candidates, X_train, Y_train, self.ComboGraph 
        else:  # randomly choose a node from the best node's neighbours
            candidate_idx = np.unique(
                np.random.RandomState(self.seed).choice(len(neighbors_of_best), 1)
            ).tolist()
            candidates = neighbors_of_best[candidate_idx]
            # Now we add these new graph structure to ComboGraph for recording purposes
            center_combonode_to_stack = np.repeat(
                np.array(best_loc).reshape(1, -1), len(neighbors_of_best), axis=0
            )
            comboedges_array_to_add = np.stack(
                (neighbors_of_best, center_combonode_to_stack), axis=1
            ).astype(int)
            comboedges_to_add = list(
                tuple(map(tuple, i)) for i in comboedges_array_to_add
            )  # Convert np.array to list of tuples
            self.ComboGraph.add_edges_from(comboedges_to_add)  # record the explored structure
            return candidates, self.X_train, self.Y_train, self.ComboGraph
