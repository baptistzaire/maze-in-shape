"""Prim's algorithm maze generator implementation.

This module implements a maze generator using a modified version of Prim's
algorithm. This approach tends to create mazes with more uniform paths and
many shorter branches compared to DFS.
"""
import random
from typing import Set, Dict, Tuple
from collections import defaultdict

from .types import Cell, Wall, Maze
from .base_generator import BaseMazeGenerator

class PrimMazeGenerator(BaseMazeGenerator):
    """Maze generator using Prim's algorithm.
    
    This implementation grows the maze outward from a starting cell, maintaining
    a frontier of potential next cells. It produces "perfect" mazes (exactly one
    path between any two cells) with characteristically uniform, shorter corridors.
    """
    
    def generate(self, grid: Set[Cell]) -> Maze:
        """Generate a maze using Prim's algorithm.
        
        Args:
            grid: Set of valid cells where the maze can be carved.
        
        Returns:
            A set of walls that have been removed to create the maze paths.
        """
        if not grid:
            return set()
        
        # Track our maze state
        in_maze: Set[Cell] = set()  # Cells already in the maze
        frontier: Dict[Cell, Cell] = {}  # frontier cell -> cell that added it
        removed_walls: Maze = set()
        
        # Process each disconnected region
        remaining_cells = grid.copy()
        
        while remaining_cells:
            # Start a new region from a random remaining cell
            start = random.choice(list(remaining_cells))
            in_maze.add(start)
            remaining_cells.remove(start)
            
            # Add initial frontier cells for this region
            for neighbor in self._get_valid_neighbors(start, grid):
                if neighbor in remaining_cells:
                    frontier[neighbor] = start
        
            # Grow this region until its frontier is empty
            while frontier:
                # Choose a random frontier cell and the cell that added it
                next_cell = random.choice(list(frontier.keys()))
                previous_cell = frontier[next_cell]
                
                # Remove the wall between them
                removed_walls.add(self._create_wall(previous_cell, next_cell))
                
                # Add the chosen cell to the maze
                in_maze.add(next_cell)
                remaining_cells.remove(next_cell)
                del frontier[next_cell]
                
                # Update frontier with new valid neighbors
                for neighbor in self._get_valid_neighbors(next_cell, grid):
                    if neighbor in remaining_cells and neighbor not in frontier:
                        frontier[neighbor] = next_cell
        
        return removed_walls
