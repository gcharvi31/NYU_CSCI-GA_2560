import sys
import os
import logging as log
from collections import defaultdict, Counter
from copy import deepcopy

ncolors2colors = {2:['R', 'G'], 3:['R', 'G', 'B'], 4:['R', 'G', 'B', 'Y']}
clr2color = {'R':'Red', 'G':'Green', 'Y':'Yellow', 'B':'Blue'}

optional = True
check = False


class DAG:
    """
    Create DAG from input adjacency list
    """
    def __init__(self, adj_list=None, nodes=None):
        self.adj_list = adj_list if adj_list else defaultdict(list) # {'a': ['a1', 'a2', 'a3'], ...}
        self.nodes = nodes
    
    def create_node_graph(self):
        for node in self.nodes:
            if node not in self.adj_list:
                self.adj_list[node] = []
        for node in self.adj_list:
            for child in self.adj_list[node]:
                self._add_edge(src=node, dest=child)
        return dict(sorted(dict(sorted(self.adj_list.items(), key=lambda x:x[1], reverse=False)).items(), key=lambda x: x[0], reverse=False))
                
                
    def _add_edge(self, src, dest):
        if dest == '':
            self.adj_list[src] = []
            return
        if dest not in self.adj_list[src]:
            self.adj_list[src].append(dest)
        if src not in self.adj_list[dest]:
            self.adj_list[dest].append(src)


def read_input(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines


def trim_label_name(name):
    return name.strip()


def is_label_name_valid(name):
    if len(name) > 32: return False
    if len([i for i in name if i in [':', '[', ']', ',']])>0:
        return False
    return True


def parse_input(filename):
    """
    Parse input text file to adjacency list
    """
    lines = read_input(filename=filename)
    lines = [l.strip('\n') for l in lines if l != '\n']  #remove extra spaces from input
        
    adj_list = dict()
    labels = set()
    
    for line in lines:
        if ":" in line:
            src, children = line.split(": ")[0], [c.strip() for c in line.split(": ")[-1].lstrip('[').rstrip(']').split(",")]
            src = trim_label_name(src)
            children = [trim_label_name(c) for c in children]
            if is_label_name_valid(src) and ([is_label_name_valid(c) for c in children] == [True for c in children]):
                adj_list[src] = children
                labels.add(src)
                labels.update(children)
            else:
                log.error(f"Invalid label names, try again")
                exit()
            
    # remove empty items from set
    labels.discard('')
    
    return adj_list, labels


def get_true_clauses(filename):
    lines = read_input(filename=filename)
    true_clauses = [l.strip('\n') for l in lines if l != '\n']  #remove extra spaces from input
    return true_clauses


def check_missing_clauses(computed_clauses, true_clauses):
    comp_ = set(tuple(row) for row in sorted(list([clause.split(', ') for clause in computed_clauses])))
    true_ = set(tuple(row) for row in sorted(list([clause.split(', ') for clause in true_clauses])))
    
    log.info(f"Clauses in true set: {len(true_)}, Clauses. in computed set: {len(comp_)}")
    log.info(f"Missing clauses: {true_.difference(comp_)}")
    log.info(f"Extra clauses computed: {comp_.difference(true_)}")


def get_true_assignments(filename):
    lines = read_input(filename=filename)
    true_assignments = [l.strip('\n') for l in lines if l != '\n']  #remove extra spaces from input
    return true_assignments


def check_wrong_assignments(computed_assignments, true_assignments):
    comp_ = set(computed_assignments)
    true_ = set(true_assignments)
    log.info(f"Missing assignments: {true_.difference(comp_)}")
    log.info(f"Wrong assignments computed: {comp_.difference(true_)}")


def write2file(l, filename):
    # open file in write mode
    with open(filename, 'w') as fp:
        for item in l:
            # write each item on a new line
            fp.write("%s\n" % item)


def graph_constraints(adj_list, ncolor):
    """
    Create CNF clauses from adjacency list by adding constraints
    Parameters
    ----------
    adj_list: dict
        Dictionary containing adjacency list representation of input
    ncolor: int
        No. of colors to be included in the constraints
    Returns
    -------
    cnf_clauses: list
        CNF clauses
    """
    colors = ncolors2colors[ncolor]
    cnf_clauses = []
    CONSTRAINT1, CONSTRAINT2, CONSTRAINT3 = [], [], []
    for node in adj_list:
        # Constraint 1: At least one color 
        constraint1 = []
        for color in colors:
            atom = f'{node}_{color}'
            constraint1.append(atom)
        
        # Constrain 2: No adjacent same colors rhs for every edge, distinct clause for each color
        constraint2 = []
        for cons in constraint1:
            for child in sorted(adj_list[node]):
                constraint2.append((f'!{cons} !{child}_{cons[-1]}'))
        
        # Constraint 3: Optional: At most one color for each vertex
        constraint3 = []
        cons_set = []
        for cons in constraint1:
            cons_set.append((cons, [c for c in constraint1 if c != cons]))
        for item in cons_set:
            cons = item[0]
            for i in item[1]:
                constraint3.append((f'!{cons} !{i}'))

        # Add constraints of a node to cnf clauses for entire graph
        cnf_clauses.extend([" ".join(constraint1)])
        cnf_clauses.extend(constraint2)
        if optional:
            cnf_clauses.extend(constraint3)

    #     CONSTRAINT1.extend([" ".join(constraint1)])
    #     CONSTRAINT2.extend(constraint2)
    #     if optional:
    #         CONSTRAINT3.extend(constraint3)
    
    # cnf_clauses.extend(CONSTRAINT1)
    # cnf_clauses.extend(CONSTRAINT2)
    # if optional:
    #     cnf_clauses.extend(CONSTRAINT3)

    return cnf_clauses


def get_atoms_and_literals_from_cnf(cnf):
    """
    Parameters
    ----------
    cnf: list
        CNF clauses
    Returns
    -------
    atoms: literals without negation
    literals: Atoms with/without negation
    """
    l = []
    for i in cnf:
        l.extend(i.split(' '))
    atoms = sorted(list(set([i.strip('!') for i in l])))
    literals = list(set([i for i in l]))
    return atoms, literals


def propagate(atom, cnf, values):
    """
    Propagate the effect of assigning atom A to be value V.
    Delete every clause where A appears with sign V
    Delete every literal where A appears with sign not V.
    """
    cnf_ = []
    for clause in cnf:
        clause_literals = clause.split(' ')
        # delete C from S
        if (atom in clause_literals and values[atom]==True) or (f'!{atom}' in clause_literals and values[atom]==False):
            pass
        # delete A from C
        elif atom in clause_literals and values[atom]==False:
            clause_ = ' '.join([i for i in clause.split(' ') if i!=atom])
            cnf_.append(clause_)
        # delete ~A from C
        elif f'!{atom}' in clause_literals and values[atom]==True:
            clause_ = ' '.join([i for i in clause.split(' ') if i!=f'!{atom}'])
            cnf_.append(clause_)
        else:
            cnf_.append(clause)
    return cnf_


def obvious_assign(literal, values):
    """
    Given a literal L with atom A, make V[A] the sign indicated by L.
    """
    if '!' in literal:
        atom = get_atom(literal)
        values[atom] = False
    else:
        atom = literal
        values[atom] = True
    log.debug(f"assigned atom {atom} as {values[atom]} through obvious assign")
    return values


def is_literal_in_cnf(literal, cnf):
    """
    Gives clause indices where a particular literal is present in cnf clause list
    """
    present = []
    for idx, clause in enumerate(cnf):
        if literal in clause:
            present.append(idx)
    return present


def get_atom(literal):
    """
    Returns atom of the literal
    """
    if '!' in literal:
        return literal[1:]
    else:
        return literal


def get_next_pureliteral(cnf):
    """
    Pureliteral: there exists a literal L in S such that the negation of L does not appear in S
    Parameters
    ----------
    cnf: list
        CNF clauses
    Returns
    -------
    literal: str
        First instance of pureliteral atom
    """
    log.debug(f"Retrieving next pure literal from cnf ...")
    # Get counts of literals in the cnf
    l = []
    for i in cnf:
        l.extend(i.split(' '))
    literals_count = dict(Counter(l))

    for clause in cnf:
        literals = clause.split(' ')
        for literal in literals:
            # check if negation of literal in remaining clauses
            if literal[0]=='!':
                literal_negation = literal[1:]
            else:
                literal_negation = '!' + literal
            if literal_negation not in literals_count.keys():
                log.debug(f"Pure literal {literal} found. Negation of this literal is not present in remaining clauses")
                return literal
    log.debug(f"No pure literal exit, continuing ...")
    return None
    

def get_next_clause_with_single_occurrence_literal(cnf):
    """
    Parameters
    ----------
    cnf: list
        CNF clauses
    Returns
    -------
    literal: str
        Unit literal from cnf
    """
    log.debug(f"Retrieving single literal clause from cnf ...")
    for clause in cnf:
        literals = clause.split(' ')
        if len(literals) == 1:
            log.debug(f"Clause with single literal: {literals[0]} found")
            return literals[0]
    log.debug(f"No clause with single literal exist, continuing ...")
    return None


def delete_literal_from_cnf(literal, cnf):
    """
    Deletes clauses from cnf containing the literal
    Parameters
    ----------
    literal: str
        Literal to check in the clauses
    cnf: list
        CNF clauses
    Returns
    -------
    cnf_: list
        CNF clauses which do not contain the literal
    """
    clauses_with_literal = []
    for clause in cnf:
        clause_literals = clause.split(' ')
        if literal in clause_literals:
            clauses_with_literal.append(clause)
    cnf_ = [clause for clause in cnf if clause not in clauses_with_literal]
    return cnf_


def check_empty_clause(cnf):
    """
    Check if any clause is empty in the cnf
    """
    for clause in cnf:
        if len(clause)==0:
            return None
    return ''


def dpl(atoms, cnf, values):
    """
    DPLL Solver
    Algorithm reference: https://cs.nyu.edu/~davise/ai/dp.txt
    Parameters
    ----------
    atoms: list of atoms in the cnf
    cnf: list of cnf clauses
    values: dictionary containing assignments of atoms, initially atom:unbound
    
    Returns
    -------
    values: dictionary containing assignments of atoms, after successful assignment atom:True/False
    None: if assignment fails
    """
    # Loop as long as there are easy cases to cherry pick 
    while(True):
        # BASE OF THE RECURSION: SUCCESS OR FAILURE 
        if len(cnf) == 0: # Success: All clauses are satisfied
            log.debug(f"Assigning remaining unbounded atoms as False: {values}")
            for a in atoms:
                if values[a] == 'unbound':
                    log.debug(f"Unbounded {a}, assigning False")
                    values[a] = False
            return values
        
        elif check_empty_clause(cnf) is None: # Failure: Some clause is unsatisfiable under V 
            return None
        
        # EASY CASES: PURE LITERAL ELIMINATION AND FORCED ASSIGNMENT
        elif get_next_pureliteral(cnf) is not None:
            pureliteral = get_next_pureliteral(cnf)
            values = obvious_assign(literal=pureliteral, values=values)
            cnf = delete_literal_from_cnf(literal=pureliteral, cnf=deepcopy(cnf))
            log.info(f"easy case: pure literal {pureliteral}; {get_atom(pureliteral)}={values[get_atom(pureliteral)]}")
            log.debug(cnf)
        
        # Forced assignment 
        elif get_next_clause_with_single_occurrence_literal(cnf) is not None:
            literal = get_next_clause_with_single_occurrence_literal(cnf=deepcopy(cnf))
            values = obvious_assign(literal=literal, values=values)
            cnf = propagate(get_atom(literal), cnf, values)
            log.info(f"easy case: unit literal {literal}; {get_atom(literal)}={values[get_atom(literal)]}")
            log.debug(cnf)
        
        # No easy cases found 
        else:
            break
    
    
    # HARD CASE: PICK SOME ATOM AND TRY EACH ASSIGNMENT IN TURN
    try:
        unbound_atom = [k for k in sorted(values.keys()) if values[k]=='unbound'][0]
    except IndexError as e:
        log.info(f"NO VALID ASSIGNMENT")
        exit()
    
    # Try one assignment 
    values[unbound_atom] = True
    log.info(f"hard case: guess {unbound_atom}=True")
    cnf1 = deepcopy(cnf)
    cnf1 = propagate(unbound_atom, cnf1, values)
    values_new = dpl(atoms, cnf1, values)
    if (values_new != None): # Found a satisfying valuation
        return values_new
    
    # If V[A] = TRUE didn't work, try V[A] = FALSE;
    values[unbound_atom] = False
    log.info(f"contradiction: backtrack guess {unbound_atom}=False")
    cnf1 = propagate(unbound_atom, cnf, values)
    return dpl(atoms, cnf1, values)
            

def dp(cnf):
    """
    Wrapper function for DPLL Solver
    """
    atoms, _ = get_atoms_and_literals_from_cnf(cnf)
    values = {k:'unbound' for k in atoms}
    for clause in cnf:
        log.info(clause)
    return dpl(atoms, cnf, values)


def convert_back(assignments):
    """
    Convert label_color:True/False assignments to label:color
    Parameters
    ----------
    assignments: dict
        Dictionary containing output of dpll label_color:True/False
    Returns
    -------
    solutions: dict
        Dictionary label:color
    """
    true_assignments = [i for i in assignments if assignments[i] is True]
    solution = {}
    for assignment in true_assignments:
        label, value = assignment.split('_')[0], clr2color[assignment.split('_')[-1]]
        solution[label]=value
    return solution


if __name__ == '__main__':

    if not os.path.exists(f"outputs"):
        os.mkdir("outputs")

    # Parse command line arguments
    if len(sys.argv) < 2:
        print(f"Insufficient arguments")
        exit()
        
    args = [i for i in sys.argv[1:]]

    args = args[::-1]
    input_file = args[0]
    file_ = input_file.split('.txt')[0]
    args.pop(0)

    ncolor = int(args[0])
    args.pop(0)
    if ncolor < 2:
        log.error(f"ncolors should be greater than 2")
        exit()
    if ncolor > 4:
        log.warning(f"ncolors defaulting to highest available ncolor=4")
        ncolor = 4

    if len(args) > 0:
        verbose = True
        log.basicConfig(
            # filename="outputs/log.txt",
            # filemode='w',
            format="%(levelname)s: %(message)s",
            level=log.INFO
                        )
    else:
        verbose = False
        log.basicConfig(format="%(levelname)s: %(message)s")

    # Converting given input file to graph adjacency list
    adj_list_1, nodes = parse_input(filename=input_file)
    graph = DAG(adj_list=adj_list_1, nodes=nodes).create_node_graph()
    log.debug(graph)
    # Converting adjacency list to cnf clauses
    clauses = graph_constraints(adj_list=graph, ncolor=ncolor)

    if check:
        true_filename = f"{input_file}.{ncolor}.dp"
        if optional:
            true_filename = f"{input_file}.{ncolor}.opt.dp"
        true_clauses = get_true_clauses(filename=true_filename)
        check_missing_clauses(true_clauses=true_clauses, computed_clauses=clauses)        
        out_filename = f"{input_file}.{ncolor}.dp"
        if optional:
            out_filename = f"{input_file}.{ncolor}.opt.dp"
        write2file(l=clauses, filename=f'outputs/{out_filename}')

    assigments = dp(cnf=clauses)
    if assigments is not None:
        log.info("Successful")
    else:
        log.info("NO VALID ASSIGNMENT")
        exit()
    solution = convert_back(assignments=assigments)
    computed_assignments = []
    for k,v in solution.items():
        computed_assignments.append(f"{k} = {v}")
        print(f"{k} = {v}")

    if check:
        true_filename = f"{file_}.{ncolor}.out"
        if optional:
            true_filename = f"{file_}.{ncolor}.opt.out"
        true_assignments = get_true_assignments(filename=true_filename)
        check_wrong_assignments(true_assignments=true_assignments, computed_assignments=computed_assignments)
        out_filename = f"{file_}.{ncolor}.out"
        if optional:
            out_filename = f"{file_}.{ncolor}.opt.out"
        write2file(l=[f'{k} = {v}' for k,v in solution.items()], filename=f'outputs/{out_filename}')
