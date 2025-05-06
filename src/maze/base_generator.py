"""Base class for maze generation algorithms.

This module provides the abstract base class that all maze generation algorithms
must implement, ensuring a consistent interface across different implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Set, Tuple

from .types import Cell, Wall, Maze

class BaseMazeGenerator(ABC):
    """Abstract base class for maze generation algorithms.
    
    All maze generation algorithms (DFS, Prim's, etc.) must inherit from this
    class and implement the abstract methods to ensure consistent behavior
    across different implementations.
    """
    
    @abstractmethod
    def generate(self, grid: Set[Cell]) -> Maze:
        """Generate a maze within the given grid cells.
        
        Args:
            grid: Set of valid cells where the maze can be carved. These are the
                 passable cells determined by the shape extraction phase.
        
        Returns:
            Maze: A set of walls that have been removed to create the maze paths.
                 Each wall is represented as a tuple of two adjacent cells.
        
        Example:
            generator = ConcreteGenerator()
            grid = {Cell((0,0)), Cell((0,1)), Cell((1,0)), Cell((1,1))}
            maze = generator.generate(grid)
        """
        raise NotImplementedError("Concrete maze generators must implement generate()")
    
    def _get_valid_neighbors(self, cell: Cell, grid: Set[Cell]) -> List[Cell]:
        """Get all valid neighboring cells from the grid.
        
        Args:
            cell: The current cell to find neighbors for.
            grid: Set of all valid cells in the maze.
        
        Returns:
            List of valid neighboring cells that exist in the grid.
        """
        row, col = cell
        potential_neighbors = [
            Cell((row-1, col)),  # North
            Cell((row+1, col)),  # South
            Cell((row, col-1)),  # West
            Cell((row, col+1)),  # East
        ]
        return [n for n in potential_neighbors if n in grid]
    
    def _create_wall(self, cell1: Cell, cell2: Cell) -> Wall:
        """Create a wall between two adjacent cells.
        
        Args:
            cell1: First cell
            cell2: Second cell
        
        Returns:
            Wall representing the connection between the cells.
        """
        # Ensure consistent wall representation by ordering cells
        if (cell1 < cell2):
            return Wall((cell1, cell2))
        return Wall((cell2, cell1))
