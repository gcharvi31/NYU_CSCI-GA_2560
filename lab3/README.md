# Markov Process Solver
A python program for a generic Markov process solver.

Implements the algorithm -
```
 𝜋 = initial policy (arbitrary)
    V = initial values (perhaps using rewards)
    for {
      V = ValueIteration(𝜋) // computes V using stationery P
      𝜋' = GreedyPolicyComputation(V) // computes new P using latest V
      if 𝜋 == 𝜋' then return 𝜋, V
      𝜋 = 𝜋'
    }
```

## File structure -  
1. `mvp.py` - main file 


### Observations about the code -
- Use `python3` for running the code
- To compare outputs use `sort my_output.txt ref_output.txt | uniq -u`


## Sample command:
`python3 mvp.py some_input.txt`

`python3 mvp.py -df 0.9 -tol 0.001 -iter 100 -min some_input.txt`

where:

flag `-df`: indicates a float discount factor to use on future rewards. If present, it sets arg_df parameter in the sovler, and defaults to 1.0 if absent; type: optional, float between [0, 1]

flag `-tol`: indicates tolerance for exiting value iteration. If present, it sets arg_tol parameter in the solver, and defaults to 0.001 if absent; type: optional, float

flag `-iter`: indicates a cutoff for value iteration. If present, it sets the arg_iter parameter in the solver, and defaults to 100 if absent; type: optional, int

flag `-min`: If present, it minimize values as costs in solver, and defaults to false which maximizes values as rewards if absent; type: optional, string

some_example.txt: Input file containing node and state information. The input file consists of 4 types of input lines: Comment lines that start with # and are ignored (as are blanklines), Rewards/costs lines of the form 'name = value' where value is an integer, Edges of the form 'name : [e1, e2, e2]' where each e# is the name of an out edge from name, Probabilities of the form 'name % p1 p2 p3' (more below). These lines may occur in any order and do not have to be grouped; type: required, string


## Requirements
python3


## References and Collaboration
1. https://github.com/TestSubjector/NYU_CSCI_GA_2560/tree/master/Lab3
2. Discussions with Anoushka Gupta (ag8733)
