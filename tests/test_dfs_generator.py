"""Tests for the DFS maze generator implementation."""
import pytest
from typing import Set

from src.maze.types import Cell, Wall, Maze
from src.maze.dfs import DFSMazeGenerator

def test_empty_grid():
    """Test generating maze with empty grid."""
    generator = DFSMazeGenerator()
    maze = generator.generate(set())
    assert len(maze) == 0

def test_single_cell():
    """Test generating maze with a single cell."""
    generator = DFSMazeGenerator()
    grid = {Cell((0, 0))}
    maze = generator.generate(grid)
    assert len(maze) == 0  # No walls to remove with single cell

def test_simple_2x2_grid():
    """Test generating maze with a 2x2 grid."""
    generator = DFSMazeGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)),
        Cell((1, 0)), Cell((1, 1))
    }
    maze = generator.generate(grid)
    
    # In a perfect maze, n cells should have n-1 removed walls
    assert len(maze) == len(grid) - 1
    
    # Verify all walls connect valid cells
    for wall in maze:
        cell1, cell2 = wall
        assert cell1 in grid
        assert cell2 in grid
        # Verify cells are adjacent
        row1, col1 = cell1
        row2, col2 = cell2
        assert abs(row1 - row2) + abs(col1 - col2) == 1

def is_fully_connected(maze: Maze, grid: Set[Cell]) -> bool:
    """Helper function to verify all cells are connected."""
    if not grid:
        return True
    
    # Use first cell as starting point
    start = next(iter(grid))
    visited = {start}
    to_visit = [start]
    
    while to_visit:
        current = to_visit.pop()
        for wall in maze:
            cell1, cell2 = wall
            if cell1 == current and cell2 not in visited:
                visited.add(cell2)
                to_visit.append(cell2)
            elif cell2 == current and cell1 not in visited:
                visited.add(cell1)
                to_visit.append(cell1)
    
    return visited == grid

def test_maze_connectivity():
    """Test that the generated maze is fully connected."""
    generator = DFSMazeGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 0)), Cell((1, 1)), Cell((1, 2)),
        Cell((2, 0)), Cell((2, 1)), Cell((2, 2))
    }
    maze = generator.generate(grid)
    
    assert is_fully_connected(maze, grid)

def test_perfect_maze():
    """Test that the generated maze is perfect (no loops)."""
    generator = DFSMazeGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 0)), Cell((1, 1)), Cell((1, 2))
    }
    maze = generator.generate(grid)
    
    # In a perfect maze with n cells, there are exactly n-1 removed walls
    assert len(maze) == len(grid) - 1
    
    # Every additional removed wall would create a loop
    assert is_fully_connected(maze, grid)
