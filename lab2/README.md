# Map/Vertex coloring via DPLL
A python program to assign colors to neighboring vertices of a graph such that no two adjacent vertices have the same color
Uses the [Davis–Putnam–Logemann–Loveland (DPLL) algorithm] (https://en.wikipedia.org/wiki/DPLL_algorithm)

Algorithm pseudo code - https://cs.nyu.edu/~davise/ai/dp.txt


## File structure -  
1. `dpll.py` - main file 


### Observations about the code -
- Use `python3` for running the code
- Output vertex color assignments are printed on the terminal
- When guessing, use the smallest lexicographic atom and guess True first
- For default unbounded assignments at the end of algorithm, False is used
- To use optional constrain, the flag *optional* in the code to be set as True; default True
- To check output cnf and assignments, the flag *check* in the code writes outputs to files


## Sample command:
`python3 dpll.py -v 2 tiny.txt`

where:

flag `-v`: indicates to print verbose output. If present, it prints - the CNF clauses that are being sent to the DPLL solver, each step taken (easy assignment or guesses) and if assignment **Successful** or **NO VALID ASSIGNMENT**; type: optional, string

2: ncolors - the number of colors to solve for.  If 2 use R, G; if 3 RGB; 4 RGBY; type: required, positive integer > 1, preferably [2, 3, 4]. If ncolors > 4, it defaults to 4; type: required, int

tiny.txt: undirected graph input file. The input text file is a subset of Lab 1 (the ':' lines), in that each line should contain a vertex and some neighbors; type: required, string


## Requirements
python3



