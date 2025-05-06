"""Maze generation module.

This module provides functionality for generating mazes within constrained shapes.
The core maze representation uses a set of removed walls between adjacent cells.

Example:
    from src.maze.types import Cell, Wall, Maze

    # Create a simple two-cell maze
    cell1, cell2 = Cell((0, 0)), Cell((0, 1))
    wall = Wall((cell1, cell2))
    maze: Maze = {wall}
"""

from .types import Cell, Wall, Maze
from .base_generator import BaseMazeGenerator
from .dfs import DFSMazeGenerator
from .prim import PrimMazeGenerator
from .kruskal import KruskalMazeGenerator
from .wilson import WilsonMazeGenerator
from .utils import (
    get_valid_neighbors,
    get_all_walls,
    get_grid_dimensions,
    is_valid_cell
)
from .factory import create_maze_generator, GENERATORS

__all__ = [
    'Cell', 'Wall', 'Maze', 'BaseMazeGenerator',
    'DFSMazeGenerator', 'PrimMazeGenerator',
    'KruskalMazeGenerator', 'WilsonMazeGenerator',
    'get_valid_neighbors', 'get_all_walls',
    'get_grid_dimensions', 'is_valid_cell',
    'create_maze_generator', 'GENERATORS'
]
