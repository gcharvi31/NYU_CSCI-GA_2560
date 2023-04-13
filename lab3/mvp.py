import sys
import logging as log

log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)


class Node:
    """
    Class Node represents a node in the graph input
    """

    def __init__(self, name):
        self.name = name
        self.node_type = None
        self.reward = None
        self.value = 0
        self.edges = []
        self.probs = []
        self.current_policy = name
        
    def __repr__(self,):
        stats = {
            "name": self.name,
            "node_type": self.node_type,
            "edges": self.edges,
            "probs": self.probs,
            "value": self.value,
            "reward": self.reward,
            "current_policy": self.current_policy
        }
        repr_ = str(stats)
        return repr_
    
    def __str__(self,):
        return f"({self.name}, {self.node_type})"
    
    def add_to_edges(self, edge):
        self.edges.append(edge)
        
    def add_to_probs(self, prob):
        self.probs.append(prob)
        
    def sum_probs(self,):
        return sum(self.probs)
    
    def print_policy(self,):
        print(f"{self.name} -> {self.current_policy}")


def read_input(filename):
    """
    Read input file
    Parameters
    ----------
    filename : str
        The name of the input file
    Returns
    -------
    lines : list
        A list of lines in the input file
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines


def clean_line(line):
    """
    Clean punctuation characters of the line
    Parameters
    ----------
    line : str
        The line to be cleaned
    Returns
    -------
    line : str
        The cleaned line
    """
    if "[" in line:
        line = line.replace("[", " [ ")
    if "]" in line:
        line = line.replace("]", " ] ")
    if ":" in line:
        line = line.replace(":", " : ")
    if "%" in line:
        line = line.replace("%", " % ")
    if "=" in line:
        line = line.replace("=", " = ")
    if "," in line:
        line = line.replace(",", " ")
    return line


def parse_input(filename):
    """
    Tokenize line, convert tokens to numbers
    Parameters
    ----------
    filename : str
        The name of the input file
    Returns
    -------
    lines_ : list
        A list of tokenized lines in the input file
    """
    # Read and clean input lines
    lines = read_input(filename=filename)
    lines = [l.strip('\n') for l in lines if l != '\n']  #remove extra spaces from input
    lines = [l for l in lines if not "#" in l]    
    # tokenize line
    lines = [clean_line(l) for l in lines]
    lines = [l.strip().split() for l in lines]
    # convert numeric tokens to numerics
    lines_ = []
    for line in lines:
        # for float values
        line = [float(item) if item.replace('.','',1).isdigit() == True else item for item in line]
        # for negative values
        line = [float(item) if (type(item)==str and item.replace('-','',1).isdigit() == True) else item for item in line]
        lines_.append(line)
    return lines_


def create_nodes_from_input(lines):
    """
    Create nodes from tokenized lines
    Parameters
    ----------
    lines : list
        A list of tokenized lines in the input file
    Returns
    -------
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object.
    """
    name2nodes = {}
    for line in lines:
        name = line[0]
        if name not in name2nodes:
            name2nodes[name] = Node(name=name)
    return name2nodes


def assign_rewards_probs_edges(lines, name2nodes):
    """
    Parse input lines and assign rewards, probabilities and edges to nodes
    Parameters
    ----------
    lines : list
        A list of tokenized lines in the input file
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object.
    Returns
    -------
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object,
        with rewards, probabilities and edges assigned.
    """
    for line in lines:
        # Reward/cost line
        if "=" in line:
            name, reward = line[0], line[-1]
            name2nodes[name].reward = reward
            name2nodes[name].value = reward
        # Probabilities line
        if "%" in line: 
            name, probs = line[0], line[line.index('%')+1:]
            for prob_val in probs:
                name2nodes[name].add_to_probs(prob_val)
        # Edges line
        if all(x in line for x in ['[', ']']): 
            name, edges = line[0], line[line.index('[')+1:line.index(']')]
            for edge in edges:
                name2nodes[name].add_to_edges(edge)
    return name2nodes



def assign_nodetype(name2nodes):
    """
    Assign node type to nodes in node dictionary
    Parameters
    ----------
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object.
    Returns
    -------
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object,
        with node types assigned.
    """
    for name in name2nodes:
        node = name2nodes[name]

        # If a node has edges but no probability entry, it is assumed to be a decision node with p=1
        if len(node.edges)!=0:
            if len(node.probs)==0:
                node.add_to_probs(1.0)
            if len(node.probs)==1:
                node.node_type = 'Decision'
        
        # If a node has edges but no reward entry, it is assumed to have a reward of 0
        if len(node.edges)!=0 and node.reward is None:
            node.reward = 0
        
        # If a node has no edges it is terminal. A probability entry for such a node is an error.
        if len(node.edges)==0:
            if len(node.probs)!=0:
                print("probability entry for Terminal node found, exiting... ")
                exit(0)
            else:
                node.node_type = 'Terminal'
        
        # A node with the same number of probabilities as edges is a chance node, with synchronized positions.        
        if len(node.edges)>0 and len(node.edges)==len(node.probs):
            node.node_type = 'Chance'
        
    
    for name in name2nodes:
        node = name2nodes[name]
        if node.node_type=='Chance':
            try:
                assert node.sum_probs()==1.0
            except:
                log.error("Chance node probabilities do not sum to 1.0, exiting...")
                exit(0)
    
    return name2nodes


def compute_new_value(node, arg_df, name2nodes):
    """
    Use Bellman update equation to compute new value for a Decision/Chance node.
    Parameters
    ----------
    node : Node
        Node for which value is to be computed
    arg_df : float
        The discount factor
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object.
    Returns
    -------
    new_value : float
        The new value of the node computed using the Bellman update equation.
    """
    new_value = 0.0
    if node.node_type=='Terminal':
        return node.reward
    else:
        new_value, edge_value_sum = 0.0, 0.0
        if node.node_type=='Chance':
            # log.debug(f"Computing value for chance node: {node.name}")
            for idx, edge in enumerate(node.edges):
                edge_value_sum += node.probs[idx]*name2nodes[edge].value
        elif node.node_type=='Decision':
            # log.debug(f"Computing value for decision node: {node.name}")
            policy, main_prob = node.current_policy, node.probs[0]
            rem_prob = (1-main_prob)/(len(node.edges) - 1) if len(node.edges)!=1 else 0.0
            for _, edge in enumerate(node.edges):
                if policy==edge:
                    edge_value_sum += main_prob*name2nodes[edge].value
                else:
                    edge_value_sum += rem_prob*name2nodes[edge].value
        new_value = node.reward + (arg_df * edge_value_sum)
        return new_value


def value_iteration_step(name2nodes, arg_df):
    """
    Perform a single step of value iteration.
    Parameters
    ----------
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object.
    arg_df : float
        The discount factor
    Returns
    -------
    max_delta : float
        The maximum change in value for any node in the graph.
    """
    max_delta = 0.0
    for name in name2nodes:
        node, value_old = name2nodes[name], name2nodes[name].value
        value_new = compute_new_value(node, arg_df=arg_df, name2nodes=name2nodes)
        node.value = value_new
        delta = abs(value_new - value_old)
        if delta > max_delta:
            max_delta = delta
    return max_delta
        

def value_iteration(name2nodes, arg_df, arg_tol, arg_iter):
    """
    ValueIteration computes a transition matrix using a fixed policy, 
    then iterates by recomputing values for each node using the previous values until either:
    no value changes by more than the 'tol' flag, or -iter iterations have taken place.
    
    Parameters
    ----------
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object.
    arg_df : float
        The discount factor.
    arg_tol : float
        The float tolerance for exiting value iteration
    arg_iter : int
        The cutoff number of iterations for the value iteration.
    Returns
    -------
    None
    """
    for _ in range(arg_iter):
        delta = value_iteration_step(name2nodes, arg_df)
        if arg_tol >= delta:
            break
            

def compute_new_policy(node, name2nodes, arg_min):
    """
    Compute a new policy for a node using the current values.
    Parameters
    ----------
    node : Node
        Node for which policy is to be computed
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object.
    arg_min : bool
        If True, the solver will minimize values as costs
    Returns
    -------
    new_policy : str
        The new policy for the node.
    """
    edge_candidate = node.current_policy
    edge_candidate_value = name2nodes[edge_candidate].value
    for _, edge in enumerate(node.edges):
        if edge == edge_candidate:
            continue
        else:
            edge_value = name2nodes[edge].value
            if (arg_min and edge_value < edge_candidate_value) or (not arg_min and edge_value > edge_candidate_value):
                edge_candidate = edge
                edge_candidate_value = edge_value
    
    return edge_candidate
    
    
def greedy_policy_computation(name2nodes, arg_min):
    """
    Compute a new policy for each node using the current values,
    if the policy changes return True, else False.
    Parameters
    ----------
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object.
    arg_min : bool
        If True, the solver will minimize values as costs, if False/defaults to false which maximizes values as rewards.
    Returns
    -------
    policy_change : bool
        True if a policy of any node has changed, False otherwise.
    """
    policy_change = False
    for name in name2nodes:
        node = name2nodes[name]
        if node.current_policy == None:
            continue
        else:
            old_policy = node.current_policy
            new_policy = compute_new_policy(node, name2nodes, arg_min)
            if old_policy != new_policy:
                policy_change = True
            node.current_policy = new_policy
    return policy_change
    
    
def markov_process_solver(name2nodes, arg_df, arg_min, arg_tol, arg_iter):
    """
    Markov process solver
    Parameters
    ----------
    name2nodes : dict
        A dictionary of nodes, where the key is the name of the node and the value is the node object.
    arg_df : float
        The discount factor.
    arg_min : bool
        If True, the solver will minimize values as costs, if False/defaults to false which maximizes values as rewards.
    arg_tol : float
        The float tolerance for exiting value iteration
    arg_iter : int
        The cutoff number of iterations for the value iteration.
    Returns
    -------
    """
    # Define initial arbitrary policy
    for name in name2nodes:
        if name2nodes[name].node_type=='Decision':
            name2nodes[name].current_policy = name2nodes[name].edges[0]
    count = 1
    while True:
        value_iteration(name2nodes=name2nodes, arg_df=arg_df, arg_tol=arg_tol, arg_iter=arg_iter)
        log.debug(f"While loop {count}")
        count += 1 
        log.debug([(node.name, node.value) for node in name2nodes.values()])
        if not greedy_policy_computation(name2nodes=name2nodes, arg_min=arg_min):
            break


def main():

    # Parse command line arguments
    if len(sys.argv) < 1:
        print(f"Insufficient arguments")
        exit()

    args = [i for i in sys.argv[1:]]

    input_filename = args[-1]
    args = args[:-1]

    if "-df" in args:
        arg_df = float(args[args.index("-df") + 1])
    else:
        arg_df = 1.0
    
    if "-min" in args:
        arg_min = True
    else:
        arg_min = False
    
    if "-tol" in args:
        arg_tol = float(args[args.index("-tol") + 1])
    else:
        arg_tol = 0.001

    if "-iter" in args:
        arg_iter = int(args[args.index("-iter") + 1])
    else:
        arg_iter = 100

    parsed_lines = parse_input(input_filename)
    name2nodes = create_nodes_from_input(parsed_lines)
    name2nodes = assign_rewards_probs_edges(lines=parsed_lines, name2nodes=name2nodes)
    log.debug([(node.name, node.value) for node in name2nodes.values()])
    name2nodes = assign_nodetype(name2nodes=name2nodes)
    log.debug([(node.name, node.node_type, node.value) for node in name2nodes.values()])
    
    markov_process_solver(name2nodes=name2nodes, arg_df=arg_df, arg_min=arg_min, arg_tol=arg_tol, arg_iter=arg_iter)
    
    # print results
    for name in sorted(name2nodes.keys()):
        if name2nodes[name].node_type=='Decision' and len(name2nodes[name].edges) > 1:
            name2nodes[name].print_policy()
    print()

    val_str = ""
    for name in sorted(name2nodes.keys()):
        val_ = round(name2nodes[name].value, 3)
        val_str += f"{name}={format(val_, '.3f')} "
    print(val_str)

if __name__ == "__main__":
    main()