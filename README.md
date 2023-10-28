# grid-domains
Code for NeurIPS 2023 paper Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal

This repository contains the grid domains, namely, Sokoban, Maze-with-teleports and Sliding tile. 

Run $pip install networkx to create the search tree graph.

Note: Depending on your pip version, the bellman loss might run or fail. Try to downgrade pip if that is the case.

Example usage:

$ cd sokoban
$ cd train
$ python3 main.py --alg astar --loss lstar
