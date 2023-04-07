## Alpha Beta Pruning
## ------------------
from collections import defaultdict
import numpy as np


def condition(alpha, beta):
    return beta <= alpha


class Node_ab:
    """Class Node represents a node in the graph and supports alpha-beta pruning"""
    def __init__(self, label, value=None, node_type='min', alpha=float("-inf"), beta=float("inf")):
        self.label = label
        self.value = value
        self.node_type = node_type
        self.children = []
        self.alpha = alpha
        self.beta = beta
        
    def __repr__(self,):
        return str(self.label)
    
    def __str__(self,):
        return f"({self.label}, {self.value}, {self.node_type}, {self.alpha}, {self.beta})"
    
    
class DAG_ab:
    """Class DAG represent the grpah input and supports alpha-beta pruning with max value cutoff"""
    def __init__(self, adj_list=None, root_label=None, root_node_type='max', n=None, v=None, label2val=None):
        self.adj_list = adj_list if adj_list else defaultdict(list) # {'a': ['a1', 'a2', 'a3'], ...}
        self.label2node = {}  # {'a': Node('a'), 'a1': Node('a1'), 'a2': Node('a2'), ...}
        self.root_label = root_label # 'a'
        self.root_node_type = root_node_type
        self.root = None
        self.n = n
        self.verbose = v
        self.label2val=label2val
        
    def __repr__(self,):
        return f"DAG({str(self.adj_list)})"
    
    def __str__(self):
        return f"DAG({str(self.adj_list)})"
    
    def create_node_graph(self):
        if self.root_label is None:
            raise ValueError("Root cannot be None before instantiating graph!")
        alpha = float('-inf')
        beta = float('inf')
        self.root = self._create_nodes(self.root_label, self.root_node_type, alpha, beta)

        
    def _create_nodes(self, root_label, root_node_type, root_alpha, root_beta):
        # root_label = 'a'
        
        # Check if root label already has a Node object
        if root_label in self.label2node:
            # print(f"root_label = {root_label} already in label2node.")
            root_node = self.label2node[root_label]
            root_node.alpha = root_alpha
            root_node.beta = root_beta
            # print(f"{root_node} old reused ^^^^^^^^^^^^^^^^^^^^^^^^^^")
        else:
            root_node = Node_ab(label=root_label,
                             value=self.label2val[root_label],
                             node_type=root_node_type,
                             alpha=root_alpha,
                             beta=root_beta)
            self.label2node[root_label] = root_node
            # print(f"{root_node} created ^^^^^^^^^^^^^^^^^^^^^^^^^^")

        if root_label in self.adj_list:
            # print(f"Present: {self.adj_list[root_label]}")
            # print(f"Rootnode before processing children: {root_node}")
            child_values = []
            for child_label in self.adj_list[root_label]:
                local_verbose = True
                # print(f"Processing child: {child_label} of node:{root_label}")
                if root_node_type == 'max':
                    child_node_type = 'min'
                if root_node_type == 'min':
                    child_node_type = 'max'
                child_node = self._create_nodes(child_label, child_node_type, root_node.alpha, root_node.beta)
                
                # set alpha and beta
                if root_node.node_type == 'max':
                    if child_node.value > root_node.alpha:
                        root_node.alpha = child_node.value
                        
                if root_node.node_type == 'min':
                    if child_node.value < root_node.beta:
                        root_node.beta = child_node.value
                
                if condition(root_node.alpha, root_node.beta):
                    print(f"pruning siblings after {child_label} of parent:{root_label}.........")
                    local_verbose = False
                    root_node.children.append(child_node)
                    if root_node.node_type == 'max':
                        if self.n and child_node.value == self.n:
                            root_node.value = child_node.value
                            choice = child_label
                            if self.verbose and local_verbose:
                                print(f"({root_node_type}){root_label} chooses {choice} for {root_node.value}")
                            return root_node
                
                    if root_node.node_type == 'min':
                        # print(f"*********** {child_node.value}, {root_node.value} ***********")
                            if self.n and child_node.value == -self.n:
                                root_node.value = child_node.value
                                choice = child_label
                                if self.verbose and local_verbose:
                                    print(f"({root_node_type}){root_label} chooses {choice} for {root_node.value}")
                                return root_node
                
                    child_values.append(child_node.value)
                    print(f"Rootnode {root_node} after processing child:{child_node}; further children (if any) of {root_node.label} will not be processed")
                    break;
                
                root_node.children.append(child_node)

                if root_node.node_type == 'max':
                    # print(f"*********** {child_node.value}, {root_node.value} ***********")
                        if self.n and child_node.value == self.n:
                            root_node.value = child_node.value
                            choice = child_label
                            if self.verbose and local_verbose:
                                print(f"({root_node_type}){root_label} chooses {choice} for {root_node.value}")
                            return root_node
                
                if root_node.node_type == 'min':
                    # print(f"*********** {child_node.value}, {root_node.value} ***********")
                        if self.n and child_node.value == -self.n:
                            root_node.value = child_node.value
                            choice = child_label
                            if self.verbose and local_verbose:
                                print(f"({root_node_type}){root_label} chooses {choice} for {root_node.value}")
                            return root_node
                
                child_values.append(child_node.value)
                
                # print(f"Rootnode {root_node} after processing child:{child_node}")

            # print(f"%%%%%%%% {root_node.children, [child.value for child in root_node.children]}")
            if root_node.node_type == 'max':
                root_node.value = max(child_values)
                # temp = child_values[::-1]
                # choice = root_node.children[len(temp) - np.argmax(temp) - 1].label
                choice = root_node.children[np.argmax(child_values)].label
                
            if root_node.node_type == 'min':
                root_node.value = min(child_values)
                # temp = child_values[::-1]
                # choice = root_node.children[len(temp) - np.argmin(temp) - 1].label
                choice = root_node.children[np.argmin(child_values)].label


            if (self.verbose and local_verbose) or (root_label == self.root_label):
                print(f"({root_node_type}){root_node.label} chooses {choice} for {root_node.value} ")        
            
        # print(f"Rootnode {root_node} after processing all its children")
        
        return root_node