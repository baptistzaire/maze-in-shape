"""Depth-First Search maze generator implementation.

This module implements a maze generator using the Depth-First Search (DFS)
algorithm with recursive backtracking. This approach tends to create mazes
with long, winding corridors.
"""
import random
from typing import Set, List, Dict
from collections import deque

from .types import Cell, Wall, Maze
from .base_generator import BaseMazeGenerator

class DFSMazeGenerator(BaseMazeGenerator):
    """Maze generator using Depth-First Search with recursive backtracking.
    
    This implementation uses an iterative DFS with a stack, rather than actual
    recursion, to avoid potential stack overflow with large mazes. It produces
    "perfect" mazes (exactly one path between any two cells) with characteristically
    long corridors.
    """
    
    def generate(self, grid: Set[Cell]) -> Maze:
        """Generate a maze using DFS with recursive backtracking.
        
        Args:
            grid: Set of valid cells where the maze can be carved.
        
        Returns:
            A set of walls that have been removed to create the maze paths.
        """
        if not grid:
            return set()
        
        # Track visited cells and removed walls
        visited: Set[Cell] = set()
        removed_walls: Maze = set()
        
        # Start from a random cell
        start = random.choice(list(grid))
        stack = deque([start])
        visited.add(start)
        
        while stack:
            current = stack[-1]  # Peek at top of stack
            
            # Get unvisited neighbors
            neighbors = self._get_valid_neighbors(current, grid)
            unvisited = [n for n in neighbors if n not in visited]
            
            if unvisited:
                # Choose random unvisited neighbor
                next_cell = random.choice(unvisited)
                
                # Remove wall between current and chosen cell
                removed_walls.add(self._create_wall(current, next_cell))
                
                # Mark as visited and add to stack
                visited.add(next_cell)
                stack.append(next_cell)
            else:
                # Backtrack - no unvisited neighbors
                stack.pop()
        
        return removed_walls
