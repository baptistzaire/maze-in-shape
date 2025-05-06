"""Maze generator factory module.

This module provides factory functionality to create maze generator instances
based on algorithm names specified in configuration.
"""
from typing import Dict, Type, Optional

from .base_generator import BaseMazeGenerator
from .dfs import DFSMazeGenerator
from .prim import PrimMazeGenerator
from .kruskal import KruskalMazeGenerator
from .wilson import WilsonMazeGenerator

# Map algorithm names to their generator classes
GENERATORS: Dict[str, Type[BaseMazeGenerator]] = {
    'dfs': DFSMazeGenerator,
    'prim': PrimMazeGenerator,
    'kruskal': KruskalMazeGenerator,
    'wilson': WilsonMazeGenerator,
}

def create_maze_generator(algorithm: str) -> BaseMazeGenerator:
    """Create a maze generator instance based on algorithm name.
    
    Args:
        algorithm: Name of the maze generation algorithm to use.
                 Must be one of: 'dfs', 'prim', 'kruskal', 'wilson'
    
    Returns:
        An instance of the specified maze generator.
    
    Raises:
        ValueError: If the specified algorithm is not supported.
    
    Example:
        generator = create_maze_generator('dfs')
        maze = generator.generate(grid)
    """
    algorithm = algorithm.lower()
    if algorithm not in GENERATORS:
        valid_algs = ", ".join(sorted(GENERATORS.keys()))
        raise ValueError(
            f"Unsupported maze algorithm: '{algorithm}'. "
            f"Must be one of: {valid_algs}"
        )
    
    return GENERATORS[algorithm]()
