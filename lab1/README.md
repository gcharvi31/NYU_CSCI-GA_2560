# MinMax Tree with Alpha-Beta Pruning and Max-value cutoff
A python program to solve a minimax game tree with alpha-beta pruning and max-value cutoff


## File structure -  
1. `basic.py` - contains the basic min-max tree solver with max-cutoff 
2. `prune.py` - contains the alpha-beta pruning tree solver with max-value cutoff 
3. `minmax.py` - main file

### Observations about the code -
- Use `python3` for running the code
- Outputs are printed on the terminal
- In case of ties in children values for setting the value of root node (based on argmin or argmax), the first child is chosen
- Condition for pruning the successor children of a node is `alpha>=beta`

## Sample command:
`python3 minmax.py -v -ab 5 min example1.txt`

where:

flag `-v`: indicates to print verbose output. If present, it prints all intermeditate choices made at the internal nodes. If absent, it prints the value and choice of the root node only; type: optional

flag `-ab`: indicates whether to use alpha-beta pruning. If present, it computes the result with alpha beta pruning algorithm. If absent, it solves the basic min-max tree only; type: optional

5: indicates to use max-value cutoff. If present, the values of all nodes are between [-number, number]; type: optional, positive integer

min/max: indicates root node type; type: required, string

example1.txt: graph input file; type: required, string


## Requirements
numpy==1.23.3
