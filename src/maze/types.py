"""Type definitions for maze generation and representation.

This module contains the core data structures used to represent mazes, including:
- Cell coordinates as (row, col) tuples
- Wall representations as pairs of adjacent cells
- Sets of removed walls that define the carved paths through the maze
- A comprehensive MazeData dictionary structure
"""
from typing import Tuple, Set, NewType, TypedDict, List, Optional

# Type aliases for clearer intent
Point = Tuple[int, int]
"""A point or cell in the maze grid, represented as a (row, col) tuple."""

Cell = NewType('Cell', Point) # Can still use NewType for semantic distinction if desired
"""A cell in the maze grid, represented as a (row, col) tuple."""

Wall = NewType('Wall', Tuple[Point, Point]) # Use Point for clarity
"""A wall between two adjacent cells, represented as a tuple of two Points."""

# While 'Maze' as a set of removed walls is useful during generation,
# the rendering and solving functions use a more comprehensive data structure.
# Define a TypedDict for this structure.
class MazeData(TypedDict):
    """
    A comprehensive dictionary containing all necessary data to represent a maze.
    """
    width: int
    height: int
    walls: Set[Tuple[Point, Point]] # Store walls as tuples of Points
    start: Point
    end: Point
    grid_mask: Optional[Set[Point]] # Optional set of valid cells

# The 'Maze' type as a set of removed walls is still relevant for generators
Maze = Set[Wall]
"""The maze structure, represented as a set of walls that have been removed.

The maze is defined by which walls have been "carved through" (removed) to create
paths. Each wall is represented by the two cells it separates.

Example:
    A maze with one path between (0,0) and (0,1) would be represented as:
    {((0,0), (0,1))}
"""
