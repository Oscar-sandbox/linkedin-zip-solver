# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 22:21:16 2025
@author: oscar
"""
from copy import deepcopy
import networkx as nx

class ZipSolver:
    def __init__(self, depth): 
        '''Constructs a Zip solver.'''
        
        # Depth of the recurssion employed by the backtracking algorithm. 
        self.depth = depth
        
        # The cells that are not yet traversed by the candidate solution 
        # form an undirected graph G. Vertices represent cells, and edges
        # represent neighboring cells with no wall between them. 
        self.G = nx.Graph()
        self.path = []
        self.last_checkpoint = (-1,-1)
        self.last_visited_checkpoint = 0
    
    def reject(self): 
        '''Rejects the candidate path as a possible solution.'''
        
        # Ensure that every cell not yet traversed by the candidate path has 
        # two vacant neighboring cells (one for the path to enter and 
        # one to exit).
        endpoints = {self.path[-1], self.last_checkpoint}
        if any(d<2 for v,d in self.G.degree if v not in endpoints): 
            return True
        
        # Ensure the order of the checkpoints is maintained. 
        head_checkpoint = self.G.nodes[self.path[-1]].get('checkpoint')
        if head_checkpoint and head_checkpoint != self.last_visited_checkpoint + 1: 
            return True
        
        # Check if the candidate path does not disconnect the remaining cells.
        if not nx.is_connected(self.G): 
            return True
    
    def accept(self): 
        '''Accepts the candidate path as a valid solution.'''
        
        # If the number of nodes remaining in G is one, this means every node 
        # except one, the one with the last checkpoint, has been traversed. 
        return self.G.order() == 1
    
    def search_next(self, v):
        '''Extend the candidate path to a new cell.'''
        
        # Update the last checkpoint reached, if necessary. 
        head_checkpoint = self.G.nodes[self.path[-1]].get('checkpoint')
        if head_checkpoint: self.last_visited_checkpoint = head_checkpoint 
        
        # Update the graph of unvisited cells and update the candidate path. 
        self.G.remove_node(self.path[-1])
        self.path.append(v)
        
    def solve_rec(self, depth): 
        '''Recursion function for solving the puzzle with backtracking.'''
        
        if self.reject(): return False
        if self.accept(): return True
        if not depth: return 
        
        # List the vacant neighbors of the head of the candidate path. 
        neighbors = list(self.G[self.path[-1]])
        
        # If only one neighbor is available, visit this node without 
        # calling the recursive function. Continue until the candidate path
        # is rejected, accepted or more neighbors are found. 
        while len(neighbors) == 1:
            self.search_next(neighbors[0])
            if self.reject(): return False
            if self.accept(): return True
            neighbors = list(self.G[self.path[-1]])
        
        # Recursively search for candidate paths, visiting the neighboring 
        # unvisited nodes of the head of the current candidate path. 
        for v in neighbors:
            branch = deepcopy(self)
            branch.search_next(v)
            if branch.solve_rec(depth-1): 
                self.path = branch.path
                return True

    def solve(self, n, checkpoints, walls=[]): 
        '''Solves the Zip puzzle given its size, a sequence of
        checkpoint locations and a list of walls between cells.'''
        
        # Construct a grid graph G, initially with all its edges. 
        self.G = nx.grid_2d_graph(n, n, periodic=False)
        self.path = [checkpoints[0]]
        self.last_checkpoint = checkpoints[-1]
        self.last_visited_checkpoint = 0
        
        # Add checkpoint information to node attributes of G. 
        for k, (i,j) in enumerate(checkpoints): 
            self.G.nodes[(i,j)]['checkpoint'] = k+1
            
        # Remove edges from G where neighboring cells have a wall between them. 
        for i1, j1, i2, j2 in walls: 
            self.G.remove_edge((i1,j1), (i2,j2))
        
        # Call the recursive function that solves the puzzle.  
        if self.solve_rec(depth=self.depth):
            return self.path