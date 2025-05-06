"""Kruskal's algorithm maze generator implementation.

This module implements a maze generator using Kruskal's algorithm to build
a random spanning tree, producing mazes with uniform randomness in their
structure.
"""
import random
from typing import Set, List, Tuple

from .types import Cell, Wall, Maze
from .base_generator import BaseMazeGenerator
from .union_find import UnionFind

class KruskalMazeGenerator(BaseMazeGenerator):
    """Maze generator using Kruskal's algorithm.
    
    This implementation builds a random spanning tree by considering walls
    in random order and removing them if they would connect different sets
    of cells. It produces mazes with very uniform randomness in their structure.
    """
    
    def _get_all_valid_walls(self, grid: Set[Cell]) -> List[Tuple[Cell, Cell]]:
        """Get all possible walls between adjacent cells in the grid.
        
        Args:
            grid: Set of valid cells where the maze can be carved.
            
        Returns:
            List of all possible walls as (cell1, cell2) pairs.
        """
        walls = []
        processed = set()  # Track processed cells to avoid duplicates
        
        for cell in grid:
            # Get valid neighbors for this cell
            neighbors = self._get_valid_neighbors(cell, grid)
            
            # Add walls between this cell and its unprocessed neighbors
            for neighbor in neighbors:
                if neighbor not in processed:
                    walls.append((cell, neighbor))
            
            processed.add(cell)
        
        return walls
    
    def generate(self, grid: Set[Cell]) -> Maze:
        """Generate a maze using Kruskal's algorithm.
        
        Args:
            grid: Set of valid cells where the maze can be carved.
            
        Returns:
            A set of walls that have been removed to create the maze paths.
        """
        if not grid:
            return set()
        
        # Initialize Union-Find data structure
        sets = UnionFind[Cell]()
        for cell in grid:
            sets.make_set(cell)
        
        # Get all possible walls and shuffle them
        walls = self._get_all_valid_walls(grid)
        random.shuffle(walls)
        
        # Track removed walls for the maze paths
        removed_walls: Maze = set()
        
        # Process walls in random order
        for cell1, cell2 in walls:
            # If cells aren't connected yet, remove wall between them
            if not sets.connected(cell1, cell2):
                removed_walls.add(self._create_wall(cell1, cell2))
                sets.union(cell1, cell2)
        
        return removed_walls
