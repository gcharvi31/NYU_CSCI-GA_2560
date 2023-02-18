import sys
from collections import defaultdict
from basic import DAG
from prune import DAG_ab

def read_input(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines


def check_input_format(input_):
    for i in input_:
        if not i.isalnum():
            print("Node label has to be alphanumeric")
            exit()
            

def check_missing_leaves(label2val, leaf_nodes, adj_list):
    for leaf in leaf_nodes:
        if leaf not in label2val:
            lis = [li for li in adj_list.values() if leaf in li]
            parent_nodes = [k for li in lis for k in adj_list if adj_list[k]==li]
            print(f"child node {leaf} of {parent_nodes} not found")
            exit()


def check_multiple_root(root_set):
    if len(root_set) > 1:
        print(f"multiple roots: {root_set}")
        exit()
            

def find_root(child2parent, adj_list):
    root_set = set()
    for node in adj_list:
        if node not in child2parent:
            root_set.add(node)
    check_multiple_root(root_set)
    return next(iter(root_set))            
            
            
def parse_input(filename):
    lines = read_input(filename=filename)
    lines = [l.strip('\n') for l in lines if l != '\n']  #remove extra spaces from input
        
    adj_list = dict()
    label2val = dict()
    child2parent = defaultdict(list)
    labels = set()
    internal_nodes = set()
    leaf_nodes = set()
    
    for line in lines:
        if ":" in line:
            src, children = line.split(": ")[0], line.split(": ")[-1].lstrip('[').rstrip(']').split(", ")
            check_input_format([src])
            check_input_format(children)
            adj_list[src] = children
            for c in children:
                child2parent[c].append(src)
            labels.add(src)
            internal_nodes.add(src)
            labels.update(children)
            label2val[src] = None

        if "=" in line:
            label_, val = line.split("=")[0], int(line.split("=")[-1])
            check_input_format([label_])
            label2val[label_] = val
    
    leaf_nodes = labels - internal_nodes
    check_missing_leaves(label2val=label2val, leaf_nodes=leaf_nodes, adj_list=adj_list)
    root = find_root(child2parent, adj_list)
            
    return adj_list, label2val, root


def main():
    args = [i for i in sys.argv[1:]]

    args = args[::-1]
    input_file = args[0]
    args.pop(0)

    root_node_type = args[0]
    args.pop(0)

    if '-ab' in args:
        ab = True
        args.remove('-ab')
    else:
        ab = False

    if '-v' in args:
        verbose = True
        args.remove('-v')
    else:
        verbose=  False
    
    if len(args) > 0:
        n = int(args[0])
    else:
        n = None

    # Read and parse input
    adj_list, label2val, root_label = parse_input(input_file)

    if ab:
        dag_ab = DAG_ab(adj_list=adj_list, root_label=root_label, root_node_type=root_node_type,
         n=n, label2val=label2val, v=verbose)
        dag_ab.create_node_graph()

    if not ab:
        dag = DAG(adj_list=adj_list, root_label=root_label, root_node_type=root_node_type,
         n=n, label2val=label2val, v=verbose)
        dag.create_node_graph()


if __name__ == "__main__":
    main()


