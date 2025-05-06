"""Tests for the Prim's algorithm maze generator implementation."""
import pytest
from typing import Set

from src.maze.types import Cell, Wall, Maze
from src.maze.prim import PrimMazeGenerator

def test_empty_grid():
    """Test generating maze with empty grid."""
    generator = PrimMazeGenerator()
    maze = generator.generate(set())
    assert len(maze) == 0

def test_single_cell():
    """Test generating maze with a single cell."""
    generator = PrimMazeGenerator()
    grid = {Cell((0, 0))}
    maze = generator.generate(grid)
    assert len(maze) == 0  # No walls to remove with single cell

def test_frontier_growth():
    """Test that maze grows properly from frontier."""
    generator = PrimMazeGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 0)), Cell((1, 1)), Cell((1, 2))
    }
    maze = generator.generate(grid)
    
    # Verify maze properties
    assert len(maze) == len(grid) - 1  # Perfect maze property
    
    # Each wall should connect adjacent cells
    for wall in maze:
        cell1, cell2 = wall
        assert cell1 in grid and cell2 in grid
        row1, col1 = cell1
        row2, col2 = cell2
        assert abs(row1 - row2) + abs(col1 - col2) == 1

def test_disconnected_regions():
    """Test handling of disconnected regions in grid."""
    generator = PrimMazeGenerator()
    # Create grid with unreachable cell
    grid = {
        Cell((0, 0)), Cell((0, 1)),
        Cell((2, 2))  # Disconnected cell
    }
    maze = generator.generate(grid)
    
    # Should only connect reachable cells
    assert len(maze) == 1  # Only one wall between (0,0) and (0,1)
    wall = list(maze)[0]
    assert wall[0][0] == wall[1][0] == 0  # Both cells in row 0

def is_fully_connected(maze: Maze, grid: Set[Cell]) -> bool:
    """Helper function to verify all cells are connected."""
    if not grid:
        return True
    
    # Start from first cell
    start = next(iter(grid))
    visited = {start}
    to_visit = [start]
    
    while to_visit:
        current = to_visit.pop()
        # Check all walls for connections
        for wall in maze:
            cell1, cell2 = wall
            if cell1 == current and cell2 not in visited:
                visited.add(cell2)
                to_visit.append(cell2)
            elif cell2 == current and cell1 not in visited:
                visited.add(cell1)
                to_visit.append(cell1)
    
    # All reachable cells should be visited
    return all(cell in visited for cell in grid if any(
        abs(cell[0] - other[0]) + abs(cell[1] - other[1]) == 1
        for other in grid if other in visited
    ))

def test_maze_connectivity():
    """Test that the generated maze is fully connected."""
    generator = PrimMazeGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 0)), Cell((1, 1)), Cell((1, 2)),
        Cell((2, 0)), Cell((2, 1)), Cell((2, 2))
    }
    maze = generator.generate(grid)
    
    assert is_fully_connected(maze, grid)
    assert len(maze) == len(grid) - 1  # Perfect maze property
