"""Tests for Wilson's algorithm maze generator implementation."""
import pytest
from typing import Set, List

from src.maze.types import Cell, Wall, Maze
from src.maze.wilson import WilsonMazeGenerator

def test_empty_grid():
    """Test generating maze with empty grid."""
    generator = WilsonMazeGenerator()
    maze = generator.generate(set())
    assert len(maze) == 0

def test_single_cell():
    """Test generating maze with a single cell."""
    generator = WilsonMazeGenerator()
    grid = {Cell((0, 0))}
    maze = generator.generate(grid)
    assert len(maze) == 0

def test_random_walk():
    """Test random walk functionality."""
    generator = WilsonMazeGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 0)), Cell((1, 1)), Cell((1, 2))
    }
    in_maze = {Cell((0, 0))}
    
    # Perform walk from bottom-right cell
    path = generator._random_walk(Cell((1, 2)), grid, in_maze)
    
    # Verify path properties
    assert path[0] == Cell((1, 2))  # Start cell
    assert path[-1] in in_maze  # Ends at maze cell
    
    # Each step should be valid
    for i in range(len(path) - 1):
        cell1, cell2 = path[i], path[i + 1]
        # Cells should be adjacent
        row1, col1 = cell1
        row2, col2 = cell2
        assert abs(row1 - row2) + abs(col1 - col2) == 1
        # Cells should be in grid
        assert cell1 in grid
        assert cell2 in grid

def test_no_loops_in_walk():
    """Test that random walks have loops erased."""
    generator = WilsonMazeGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 0)), Cell((1, 1)), Cell((1, 2))
    }
    in_maze = {Cell((0, 0))}
    
    path = generator._random_walk(Cell((1, 2)), grid, in_maze)
    
    # Verify no cell appears twice in path
    seen = set()
    for cell in path:
        assert cell not in seen, "Loop detected in path"
        seen.add(cell)

def is_fully_connected(maze: Maze, grid: Set[Cell]) -> bool:
    """Helper function to verify all cells are connected."""
    if not grid:
        return True
    
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
    generator = WilsonMazeGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 0)), Cell((1, 1)), Cell((1, 2))
    }
    maze = generator.generate(grid)
    
    assert is_fully_connected(maze, grid)
    assert len(maze) == len(grid) - 1  # Perfect maze property

def test_perfect_maze():
    """Test that the generated maze is perfect (no loops)."""
    generator = WilsonMazeGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 0)), Cell((1, 1)), Cell((1, 2))
    }
    maze = generator.generate(grid)
    
    # In a perfect maze with n cells, there are exactly n-1 removed walls
    assert len(maze) == len(grid) - 1

def test_disconnected_regions():
    """Test handling of disconnected regions in grid."""
    generator = WilsonMazeGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)),
        Cell((2, 2))  # Disconnected cell
    }
    maze = generator.generate(grid)
    
    # Should only connect adjacent cells
    assert len(maze) == 1  # Only one wall between (0,0) and (0,1)
    wall = list(maze)[0]
    assert wall[0][0] == wall[1][0] == 0  # Both cells in row 0
