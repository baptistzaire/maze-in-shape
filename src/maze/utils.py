"""Utility functions for maze generation.

This module provides common functionality shared across different maze generation
algorithms, such as finding valid neighbors and grid operations.
"""
from typing import Set, List, Tuple

from .types import Cell

def get_valid_neighbors(grid: Set[Cell], row: int, col: int) -> List[Cell]:
    """Get valid neighboring cells from a grid position.
    
    Args:
        grid: Set of valid/passable cells in the maze
        row: Row coordinate to find neighbors for
        col: Column coordinate to find neighbors for
        
    Returns:
        List of valid neighboring cells that exist in the grid.
        Neighbors are only considered valid if they are in the grid
        and are passable (i.e., exist in the grid set).
    
    Example:
        grid = {Cell((0,0)), Cell((0,1)), Cell((1,0))}
        neighbors = get_valid_neighbors(grid, 0, 0)
        # Returns [Cell((0,1)), Cell((1,0))]
    """
    potential_neighbors = [
        Cell((row-1, col)),  # North
        Cell((row+1, col)),  # South
        Cell((row, col-1)),  # West
        Cell((row, col+1)),  # East
    ]
    return [n for n in potential_neighbors if n in grid]

def get_all_walls(grid: Set[Cell]) -> List[Tuple[Cell, Cell]]:
    """Get all possible walls between adjacent cells in a grid.
    
    Args:
        grid: Set of valid cells
        
    Returns:
        List of walls as (cell1, cell2) pairs, where cells are adjacent.
        Each wall is returned exactly once, with cells ordered to ensure
        consistent representation.
    
    Example:
        grid = {Cell((0,0)), Cell((0,1))}
        walls = get_all_walls(grid)
        # Returns [(Cell((0,0)), Cell((0,1)))]
    """
    walls = []
    processed = set()
    
    for cell in grid:
        row, col = cell
        neighbors = get_valid_neighbors(grid, row, col)
        
        # Add walls between this cell and unprocessed neighbors
        for neighbor in neighbors:
            if neighbor not in processed:
                # Order cells to ensure consistent wall representation
                if cell < neighbor:
                    walls.append((cell, neighbor))
                else:
                    walls.append((neighbor, cell))
        
        processed.add(cell)
    
    return walls

def get_grid_dimensions(grid: Set[Cell]) -> Tuple[int, int]:
    """Get the dimensions of the grid.
    
    Args:
        grid: Set of valid cells
        
    Returns:
        Tuple of (rows, cols) indicating the grid dimensions.
        Returns (0, 0) for empty grid.
    
    Example:
        grid = {Cell((0,0)), Cell((0,1)), Cell((1,0))}
        rows, cols = get_grid_dimensions(grid)
        # Returns (2, 2)
    """
    if not grid:
        return (0, 0)
    
    max_row = max(cell[0] for cell in grid)
    max_col = max(cell[1] for cell in grid)
    return (max_row + 1, max_col + 1)

def is_valid_cell(grid: Set[Cell], row: int, col: int) -> bool:
    """Check if a cell position is valid within the grid.
    
    Args:
        grid: Set of valid cells
        row: Row coordinate to check
        col: Column coordinate to check
        
    Returns:
        True if the cell exists in the grid (is passable),
        False otherwise.
    
    Example:
        grid = {Cell((0,0)), Cell((0,1))}
        is_valid = is_valid_cell(grid, 0, 0)  # Returns True
        is_valid = is_valid_cell(grid, 1, 1)  # Returns False
    """
    return Cell((row, col)) in grid
