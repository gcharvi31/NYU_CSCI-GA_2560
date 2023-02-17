from collections import defaultdict
import numpy as np

class Node:
    """Class Node represents a node in the graph input"""

    def __init__(self, label, value=None, node_type='min'):
        self.label = label
        self.value = value
        self.node_type = node_type
        self.children = []
        
    def __repr__(self,):
        return str(self.label)
    
    def __str__(self,):
        return f"({self.name}, {self.value}, {self.node_type})"
    
    
class DAG:
    """Class DAG represents the graph input and supports max-value cutoff"""

    def __init__(self, adj_list=None, root_label=None, root_node_type='max', n=None, label2val=None, v=None):
        self.adj_list = adj_list if adj_list else defaultdict(list) # {'a': ['a1', 'a2', 'a3'], ...}
        self.label2node = {}  # {'a': Node('a'), 'a1': Node('a1'), 'a2': Node('a2'), ...}
        self.root_label = root_label # 'a'
        self.root_node_type = root_node_type
        self.root = None
        self.n = n
        self.verbose = v
        self.label2val = label2val
        
    def __repr__(self,):
        return f"DAG({str(self.adj_list)})"
    
    def __str__(self):
        return f"DAG({str(self.adj_list)})"
    
    def create_node_graph(self):
        if self.root_label is None:
            raise ValueError("Root cannot be None before instantiating graph!")
        
        self.root = self._create_nodes(self.root_label, self.root_node_type)
        if not self.verbose:
            res = max([child.value for child in self.root.children]) if self.root.node_type == 'max' \
                else min([child.value for child in self.root.children])
            choice = self.root.children[np.argmax([child.value for child in self.root.children])].label \
                if self.root.node_type == 'max' \
                    else self.root.children[np.argmin([child.value for child in self.root.children])].label
            print(f"({self.root.node_type}){self.root.label} chooses {choice} for {self.root.value}")

        
    def _create_nodes(self, root_label, root_node_type):
        # root_label = 'a'
        
        # Check if root label already has a Node object
        if root_label in self.label2node:
#             print(f"root_label = {root_label} already in label2node.")
            root_node = self.label2node[root_label]
        else:
#             print(f"root_label = {root_label} not in label2node.")
            root_node = Node(label=root_label, value=self.label2val[root_label], node_type=root_node_type)
            self.label2node[root_label] = root_node
            
        # root_node = Node('a')
#         print(self.label2node)
        
        child_values = []
        if root_label in self.adj_list:
#             print(f"Present: {self.adj_list[root_label]}")
            
            for child_label in self.adj_list[root_label]:
                if root_node_type == 'max':
                    child_node_type = 'min'
                if root_node_type == 'min':
                    child_node_type = 'max'
                child_node = self._create_nodes(child_label, child_node_type)
                if self.n and root_node_type == 'max' and child_node.value == self.n:
                    root_node.value = self.n
                    choice = child_node.label
                    if self.verbose:
                        print(f"({root_node_type}){root_label} chooses {choice} for {root_node.value}")
                    return root_node
                if self.n and root_node_type == 'min' and child_node.value == -self.n:
                    root_node.value = -self.n
                    choice = child_node.label
                    if self.verbose:
                        print(f"({root_node_type}){root_label} chooses {choice} for {root_node.value}")
                    return root_node
                child_values.append(child_node.value)
                root_node.children.append(child_node)
            

            root_node.value = max(child_values) if root_node.node_type == 'max' else min(child_values)
            choice = self.adj_list[root_label][np.argmax(child_values)] if root_node.node_type == 'max' else self.adj_list[root_label][np.argmin(child_values)]
            if self.verbose:
                print(f"({root_node_type}){root_label} chooses {choice} for {root_node.value}")
                
        return root_node